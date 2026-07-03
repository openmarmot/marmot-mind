"""
Simple cron jobs for Marmot.

Cron jobs are loaded from server/code/cron.json (optional; copy cron.json.example to get started).
Format (JSON array of simple objects). Only "schedule" + "prompt" are required.
Supported fields per job: "schedule", "prompt", optional "id", "enabled" (bool, defaults true), "comment" (ignored).

Each job's prompt is sent (internally) to the LLM with full tool access (ReAct). The final response text
is queued via queue_proactive_message() by the agent. Last execution time per job is persisted in
cron_state.json and used to avoid duplicate runs for the same time slot (deduped at minute granularity).
"""

import os
import json
import datetime
import threading
import time

# Config/data files live at server/code/ level (same as before the cron/ package was extracted)
# so that "copy cron.json.example -> cron.json" instructions and existing setups keep working.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRON_PATH = os.path.join(_CODE_DIR, "cron.json")
CRON_STATE_PATH = os.path.join(_CODE_DIR, "cron_state.json")

cron_jobs = []  # list of {"id": str, "schedule": str, "prompt": str, "enabled": bool, "last_run": datetime|None}


def _load_cron_state() -> dict:
    """Return {job_id: isoformat str} from disk."""
    if not os.path.exists(CRON_STATE_PATH):
        return {}
    try:
        with open(CRON_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print("Warning: could not load cron_state.json:", e)
        return {}


def _save_cron_last_run(job_id: str, when: datetime.datetime) -> None:
    state = _load_cron_state()
    state[job_id] = when.isoformat()
    try:
        with open(CRON_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except Exception as e:
        print("Warning: could not save cron_state.json:", e)


def _cron_field_values(field: str, min_val: int, max_val: int) -> set:
    """Expand cron field like '*', '5', '1,3', '*/15', '9-17', '1-10/2' into set of ints."""
    values = set()
    if not field or field == "*":
        return set(range(min_val, max_val + 1))
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            base, st = part.split("/", 1)
            try:
                step = max(1, int(st))
            except Exception:
                step = 1
            part = base
        if part == "*":
            start, end = min_val, max_val
        elif "-" in part:
            try:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
            except Exception:
                continue
        else:
            try:
                start = end = int(part)
            except Exception:
                continue
        for v in range(start, end + 1, step):
            if min_val <= v <= max_val:
                values.add(v)
    return values


def cron_due(schedule: str, dt: datetime.datetime) -> bool:
    """True if 5-field cron schedule matches dt (uses local time, minute resolution).

    Day matching follows classic cron "OR" rule: when both dom and dow are restricted (not *),
    the job runs if *either* the day-of-month *or* the day-of-week matches.
    """
    try:
        parts = [p.strip() for p in (schedule or "").split()]
        if len(parts) != 5:
            return False
        minute_f, hour_f, dom_f, month_f, dow_f = parts

        if dt.minute not in _cron_field_values(minute_f, 0, 59):
            return False
        if dt.hour not in _cron_field_values(hour_f, 0, 23):
            return False
        if dt.month not in _cron_field_values(month_f, 1, 12):
            return False

        doms = _cron_field_values(dom_f, 1, 31)
        dom_match = dt.day in doms
        dom_restricted = (dom_f != "*")

        # DOW: cron 0/7=Sun, 1=Mon..6=Sat; datetime.weekday Mon=0..Sun=6
        dows_raw = _cron_field_values(dow_f, 0, 7)
        dows = {0 if d == 7 else d for d in dows_raw}
        py_wd = dt.weekday()
        cron_wd = (py_wd + 1) % 7
        dow_match = (cron_wd in dows) if dows else True
        dow_restricted = (dow_f != "*")

        # Classic cron: when *both* dom and dow are restricted (neither is "*"), match if either matches (OR).
        # Otherwise require the (effective) matches (unrestricted sides always match because their set is full range).
        if dom_restricted and dow_restricted:
            day_ok = dom_match or dow_match
        else:
            day_ok = dom_match and dow_match
        if not day_ok:
            return False
        return True
    except Exception:
        return False


def load_cron_jobs():
    """Load cron jobs from CRON_PATH into the global cron_jobs list.

    Safe to call multiple times (clears and repopulates the *same* list object so that
    "from cron import cron_jobs" bindings elsewhere stay valid).
    """
    cron_jobs.clear()
    if not os.path.exists(CRON_PATH):
        if os.path.exists(CRON_PATH + ".example"):
            print("   (Cron enabled: copy cron.json.example -> cron.json to schedule prompt jobs)")
        return
    try:
        with open(CRON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Warning: cron.json must be a JSON array of {schedule, prompt} objects")
            return
        saved_runs = _load_cron_state()
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            # "comment" (and any other extra keys) are allowed for human notes and are deliberately ignored.
            comment = entry.get("comment")  # optional human-readable note only

            # "enabled" defaults to true so old configs keep working. Accepts bool or common string forms.
            enabled = entry.get("enabled", True)
            if isinstance(enabled, str):
                enabled = enabled.lower() not in ("0", "false", "no", "off", "disabled")
            enabled = bool(enabled)

            sched = str(entry.get("schedule", "")).strip()
            prompt = str(entry.get("prompt", "")).strip()
            if not sched or not prompt:
                continue
            jid = str(entry.get("id") or f"{sched}:{i}")
            last_run = None
            saved = saved_runs.get(jid)
            if saved:
                try:
                    last_run = datetime.datetime.fromisoformat(saved)
                except Exception:
                    pass
            cron_jobs.append({
                "id": jid,
                "schedule": sched,
                "prompt": prompt,
                "enabled": enabled,
                "last_run": last_run
            })
        if cron_jobs:
            enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
            schedules = ", ".join(j["schedule"] for j in enabled_jobs)
            total = len(cron_jobs)
            if len(enabled_jobs) < total:
                print(f"⏰ Loaded {len(enabled_jobs)}/{total} cron job(s) ({total - len(enabled_jobs)} disabled): {schedules}")
            else:
                print(f"⏰ Loaded {total} cron job(s): {schedules}")
    except Exception as e:
        print("Warning: could not load cron.json:", e)


def start_cron_scheduler(run_agent=None):
    """Start a daemon thread that periodically checks cron_jobs and fires any that are due.

    Due jobs run their prompt through the provided LLM agent runner (internal=True so history stays clean)
    and the resulting text is queued as a proactive message (which will be spoken + added to
    conversation on delivery, exactly like other proactive messages).

    Args:
        run_agent: Callable like process_with_llm(prompt, internal=True) -> status_str.
                   Must be supplied; this keeps the cron module decoupled from server internals.
    """
    if run_agent is None:
        raise RuntimeError("start_cron_scheduler(run_agent=...) is required (pass your process_with_llm)")

    enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
    if not enabled_jobs:
        return

    def _cron_loop():
        while True:
            try:
                time.sleep(30)  # minute-granularity crons are well served by 30s checks
                now = datetime.datetime.now()
                for job in cron_jobs:
                    if not job.get("enabled", True):
                        continue
                    sched = job.get("schedule", "")
                    prompt = job.get("prompt", "")
                    if not sched or not prompt:
                        continue
                    if not cron_due(sched, now):
                        continue
                    # Use minute slot for "already executed this occurrence?"
                    slot = now.replace(second=0, microsecond=0)
                    lr = job.get("last_run")
                    if lr is not None:
                        if lr.replace(second=0, microsecond=0) == slot:
                            continue
                    job["last_run"] = now
                    _save_cron_last_run(job["id"], now)
                    print(f"\n⏰ Cron fired [{sched}]: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
                    try:
                        # The agent loop will queue any speak() calls itself. No need to take a return value.
                        status = run_agent(prompt, internal=True)
                        print(f"🐹 Cron agent status: {status}")
                    except Exception as ex:
                        print("Cron job failed:", ex)
            except Exception as e:
                print("Cron scheduler error (retrying):", e)
                time.sleep(10)

    t = threading.Thread(target=_cron_loop, daemon=True, name="marmot-cron")
    t.start()
    print(f"⏰ Cron scheduler started for {len(enabled_jobs)} job(s)")
