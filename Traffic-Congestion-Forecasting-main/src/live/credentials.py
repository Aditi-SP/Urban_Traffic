"""
Where the TomTom key comes from, and WHY it is missing when it is missing. The key's value is only ever returned to the provider
client; every report, hint and message produced here is safe to show: it states whether a key is configured and where it was found,
never the value (not even its length).

How the dashboard finds the key (in this order; the first usable one wins)
  1. the environment variable TOMTOM_API_KEY of the SERVER process (the python process that runs Streamlit);
  2. TOMTOM_API_KEY in .streamlit/secrets.toml of the project, then in ~/.streamlit/secrets.toml.
Nothing else: the launcher does not change it and the browser never sees it.

Why "I set the variable but it is not configured" happens, and what this module reports about it
  * An environment variable is copied into a process when the process STARTS. A variable set afterwards, in another terminal window or tab,
    or as a Windows user variable (setx / SetEnvironmentVariable) is NOT seen by a dashboard that is already running, nor by terminals that
    were opened earlier. VS Code's integrated terminals are a common case: they keep the environment VS Code itself started with.
    The report therefore says when the value exists in the Windows user/machine environment but not in this process.
  * A typo in the name (TOMTOM_KEY, TOM_TOM_API_KEY) or a value wrapped in quotes or with spaces. Names that look similar are listed (names
    only); surrounding quotes and spaces are removed, and a value that still cannot be a key is reported as unusable.
  * A secrets file that is not valid TOML, or has no TOMTOM_API_KEY entry, or is in the wrong folder.

Windows settings: only the PRESENCE of the key is ever read from the Windows environment (for the hint). Two non-secret settings
(LIVE_CACHE_TTL_S, LIVE_MAX_REQUESTS) can also be compared, so the report can say when the dashboard runs with different values than
the ones saved for the Windows user. Nothing here changes the process environment: the launcher (run_dashboard.ps1) does that.
"""
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_KEY = "TOMTOM_API_KEY"
SETTINGS = ("LIVE_CACHE_TTL_S", "LIVE_MAX_REQUESTS")
ROOT = Path(__file__).resolve().parent.parent.parent


def process_start_time():
    """(unix_time, exact): when THIS operating-system process was created (the python process that runs Streamlit), from the operating system.

    Importing this module is not the start of the process (a module can be imported, or re-imported after a source change, long after it),
    so the time is asked from the OS: GetProcessTimes on Windows, /proc on Linux. If the OS cannot say, the time of this import is returned
    with exact=False, and Diagnostics words it as "loaded", not "started"."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            created, _exit, _kernel, _user = (wintypes.FILETIME() for _ in range(4))
            if k32.GetProcessTimes(k32.GetCurrentProcess(), *map(ctypes.byref, (created, _exit, _kernel, _user))):
                ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime       # 100 ns since 1601-01-01 (UTC)
                return (ticks - 116444736000000000) / 1e7, True
        elif sys.platform.startswith("linux"):
            fields = Path("/proc/self/stat").read_bytes().rsplit(b")", 1)[1].split()
            start_ticks = int(fields[19])                                            # field 22 of /proc/<pid>/stat
            boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_bytes().splitlines() if line.startswith(b"btime"))
            return boot + start_ticks / os.sysconf("SC_CLK_TCK"), True
    except Exception:
        pass
    return time.time(), False


STARTED_AT, STARTED_AT_EXACT = process_start_time()   # computed once at import and never again: the value is fixed for the life of the process
_MACHINE_ENV = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"


def normalise(value):
    """(usable_key_or_None, reason_or_None). Surrounding whitespace and one pair of matching quotes are removed."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "the value is not text"
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    if not v:
        return None, "the value is empty"
    if any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in v):
        return None, "the value contains spaces or characters that cannot be part of a key"
    return v, None


def windows_env_scope(name):
    """'user', 'machine' or None: where a NON-EMPTY Windows environment variable with this name is saved. Presence only; no value is returned."""
    return _windows_lookup(name)[0]


def windows_env_value(name):
    """The saved Windows value of a NON-SECRET setting (LIVE_CACHE_TTL_S, LIVE_MAX_REQUESTS). Refuses anything else."""
    if name not in SETTINGS:
        raise ValueError("only non-secret settings can be read")
    return _windows_lookup(name)[1]


def _windows_lookup(name):
    if os.name != "nt":
        return None, None
    try:
        import winreg
    except ImportError:
        return None, None
    for scope, root, sub in (("user", winreg.HKEY_CURRENT_USER, "Environment"), ("machine", winreg.HKEY_LOCAL_MACHINE, _MACHINE_ENV)):
        try:
            with winreg.OpenKey(root, sub) as k:
                v, _ = winreg.QueryValueEx(k, name)
        except OSError:
            continue
        if isinstance(v, str) and v.strip():
            return scope, v
    return None, None


@dataclass
class Credentials:
    key: object = field(default=None, repr=False)   # the value: NEVER print or log it
    source: object = None                           # "environment variable" / "secrets file" / None
    problems: list = field(default_factory=list)    # why a configured value is unusable (safe text)
    hints: list = field(default_factory=list)       # what to do (safe text)
    similar_names: list = field(default_factory=list)  # environment variable NAMES that look like the key's name
    secrets_files: list = field(default_factory=list)  # [(label, exists)]
    windows_scope: object = None                       # "user"/"machine" when a key is saved in Windows but this process does not have it

    @property
    def configured(self) -> bool:
        return self.key is not None

    def __str__(self):
        return f"Credentials(configured={self.configured}, source={self.source!r})"

    __repr__ = __str__

    def hint_text(self) -> str:
        """One short paragraph for a banner or a status card; empty when there is nothing useful to add."""
        return " ".join(self.problems + self.hints)

    def short_hint(self, limit=150) -> str:
        """The single most useful sentence about why the key is missing, short enough for a banner (the full reasons: hint_text())."""
        if self.configured:
            return ""
        if self.windows_scope:
            return "Saved in Windows but not loaded here: restart with `.\\run_dashboard.ps1`."
        if self.problems:
            text = self.problems[0]
        elif self.similar_names:
            text = f"A variable named {self.similar_names[0]} exists, but only TOMTOM_API_KEY is used: rename it."
        else:
            return ""
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"

    def report_lines(self) -> list:
        """Everything a person needs to see why the key is (not) configured, without the key."""
        out = [f"TomTom key: configured (source: {self.source}). The value is not shown." if self.configured
               else "TomTom key: NOT configured for this dashboard process."]
        out.append("Looked in: the TOMTOM_API_KEY environment variable of this process"
                   + "".join(f"; {label}: {'file found' if exists else 'no file'}" for label, exists in self.secrets_files) + ".")
        out += self.problems
        out += self.hints
        return out


def load(project_root=None, home=None, environ=None) -> Credentials:
    """Read the configuration of THIS process (the arguments exist for tests). Never raises."""
    env = os.environ if environ is None else environ
    root, home = Path(project_root) if project_root else ROOT, Path(home) if home else Path.home()
    c = Credentials()
    c.similar_names = sorted(n for n in env if "TOM" in n.upper() and n != ENV_KEY)
    raw = env.get(ENV_KEY)
    if raw is not None:
        key, why = normalise(raw)
        if key:
            c.key, c.source = key, "environment variable"
            if raw != key:
                c.hints.append("The environment variable had surrounding spaces or quotes; they are ignored.")
        else:
            c.problems.append(f"The environment variable TOMTOM_API_KEY exists but cannot be used: {why}.")
    for label, path in (("project secrets file .streamlit/secrets.toml", root / ".streamlit" / "secrets.toml"),
                        ("home secrets file ~/.streamlit/secrets.toml", home / ".streamlit" / "secrets.toml")):
        exists = path.is_file()
        c.secrets_files.append((label, exists))
        if c.key is not None or not exists:
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as e:
            c.problems.append(f"The {label} could not be read ({type(e).__name__}: {str(e)[:80]}). It must contain a line like TOMTOM_API_KEY = \"your-key\".")
            continue
        if ENV_KEY not in data:
            near = sorted(k for k in data if isinstance(k, str) and "TOM" in k.upper())
            c.problems.append(f"The {label} has no TOMTOM_API_KEY entry" + (f" (it has: {', '.join(near)})" if near else "") + ".")
            continue
        key, why = normalise(data[ENV_KEY])
        if key:
            c.key, c.source = key, "secrets file"
        else:
            c.problems.append(f"The TOMTOM_API_KEY entry in the {label} cannot be used: {why}.")
    if c.key is None and c.similar_names:
        c.hints.append("Environment variables with a similar name exist (" + ", ".join(c.similar_names) + "), but only TOMTOM_API_KEY is used: "
                       "rename the variable to exactly TOMTOM_API_KEY.")
    if c.key is None and environ is None:
        scope = windows_env_scope(ENV_KEY)
        c.windows_scope = scope
        if scope:
            c.hints.append(f"A TOMTOM_API_KEY is saved in your Windows {scope} environment, but THIS dashboard process does not have it (it was started "
                           "before the variable was saved, or from a terminal that was opened earlier, for example inside VS Code). "
                           "Stop the dashboard and start it again with .\\run_dashboard.ps1, which picks the saved value up.")
        elif not c.problems and not c.similar_names:
            c.hints.append("No TOMTOM_API_KEY was found in this process, in the project or home secrets file, or in your saved Windows environment.")
    return c


def settings_lines(effective: dict) -> list:
    """Safe lines about the two non-secret settings: the value this process uses and whether the Windows saved value differs."""
    out = []
    for name in SETTINGS:
        in_use = effective.get(name)
        saved = windows_env_value(name)
        line = f"{name} in use by this dashboard: {in_use}"
        if saved is not None and str(saved).strip() != str(in_use):
            line += (f". WARNING: your saved Windows value is {saved.strip()}, but this process was started without it (or before it was saved); "
                     "restart the dashboard with .\\run_dashboard.ps1 so it is applied.")
        out.append(line)
    return out
