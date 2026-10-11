# ANode
Node based realtime audio processing

## setup

Create the complete development environment with a single command
```bash
conda env create -f environment.yml
```

Activate the new environment
```bash
conda activate anode-dev
```
## compile and install
This command does two magic things:
 - Triggers scikit-build-core, which runs CMake to compile C++ code.
 - Installs Python package in a way that any changes to the .py
    files are immediately reflected without needing to reinstall.

```bash
pip install -e . -v
```

## run
```bash
python main.py
```

## test
Always run with the environment activated — activation puts `Library\bin`
on `PATH`, which native-library discovery depends on (without it, e.g.
`import soundfile` fails even though `libsndfile` is installed).
```bash
conda activate anode-dev
python -m pytest tests/ -q
```

## Supertonic 3 voice preview (offline TTS)

Two Offline nodes synthesize speech locally (CPU, no network after setup):

* **Supertonic Voice Designer** — pick a preset (M1–M5/F1–F5) or import a
  Supertonic 3 voice-style JSON; **Apply voice** publishes a unique kept
  `voice_style.json`.
* **Supertonic 3 TTS** — type text (or wire `text_uri`), resolve the voice
  via wired `voice_style_uri` → `voice_style_file` → preset, then
  **Generate speech** keeps a WAV and pulses `done` for SamplePlayer.

Setup (once):

```bash
conda activate anode-dev
python -m pip install "git+https://github.com/supertone-oss-archive/supertonic-py.git@df0f9686dac7fbbde391b759e2ee5286a3737622"
python tools/supertonic3/fetch.py
```

Open `patches/supertonic3_voice_preview.json`: Designer `voice_style` →
TTS `voice_style_uri`, Designer `done` → TTS `trigger_in`, TTS `audio` →
SamplePlayer `uri_in`, TTS `done` → SamplePlayer `trigger_in`,
SamplePlayer → output. With the engine running, Apply previews the
selected voice using the TTS test sentence; with the engine stopped,
Apply still saves the profile — press Generate manually. Model weights
are under the OpenRAIL-M license (see `models/Supertonic3/LICENSE` for
its terms); see `tools/supertonic3/README.md`.

## remove the environment
```bash
conda deactivate
conda env remove --name anode-dev
```
