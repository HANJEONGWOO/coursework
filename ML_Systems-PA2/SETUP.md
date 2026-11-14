# One-time simulator setup

You need **Python 3.10 or newer** and Linux, for example Ubuntu 22.04 LTS or
newer (including Ubuntu in WSL). Check with `python --version`. Native ASTRA
experiments run on a **Linux CPU**. You do not
need a physical GPU, CUDA, a GPU rental, or a Docker container.

Extract the whole assignment ZIP. Open a terminal in the folder that contains
`run.py` and `student.py`.

If `python` is not found but `python3` is available, run
`alias python=python3` once in the current terminal. Then run
`python --version` and make sure it reports Python 3.10 or newer. The alias
lasts until that terminal is closed; repeat it in a new terminal if needed.

## On the course Linux host

If the course ASTRA installation is already present, the runner finds it.
Otherwise, install it once:

```bash
python setup.py
python run.py
```

Setup builds the pinned simulator and the supplied event-interface patches.
It needs internet access and may take several minutes. Build messages are
saved in `setup.log`. Later experiments need only `python run.py`.
There is no environment-activation command and no installation path to enter.

## On Ubuntu or Windows with Ubuntu/WSL

If the Linux build tools are missing, run:

```bash
sudo apt install build-essential cmake git python3 python3-venv python-is-python3
python setup.py
python run.py
```

If Ubuntu reports that package lists are missing or stale, run `sudo apt update`
and repeat the install command. If you do not have administrator access, ask
course staff to prepare the Linux host.

On Windows, run these commands inside Ubuntu/Windows Subsystem for Linux (WSL),
not PowerShell. If WSL is not installed, install it with `wsl --install` from an
administrator PowerShell, restart if prompted, then open Ubuntu. Keep the
assignment folder accessible from that Linux terminal. The Windows Python
command can perform `python run.py --check`, but cannot execute the Linux
ASTRA binary. macOS users should use the course Linux host or a Linux VM.

## If setup fails

- **Missing build tool:** install the listed package, then run setup again.
- **Download or build error:** open `setup.log`. Keep that log when asking for help.
- **Old simulator version:** run `python setup.py` to build the matching version.
- **Missing assignment file:** extract the whole ZIP again into a new folder.

Setup reuses a verified installation. It checks the pinned source revisions,
course event interface, and executable checksum. It does not download an
unrelated binary or silently replace ASTRA with an estimate.

After setup, edit `student.py` and use `python run.py`. Results appear in
`results/latest/index.html`; open that file in a browser.
