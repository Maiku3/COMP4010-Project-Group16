# COMP4010-Project-Group16
Final Project for COMP 4010 Reinforcement Learning

Python 3.10
Box2D error fix:
Do this exactly:

1) # Remove the Python swig from your venv
source .venv/bin/activate
pip uninstall -y swig
# If a stub is still there, remove it and refresh shell caches
rm -f "$VIRTUAL_ENV/bin/swig"
hash -r

2) # Install system SWIG (Homebrew) and confirm PATH
# macOS Intel: usually /usr/local; Apple Silicon: /opt/homebrew
brew install swig
which swig         # expect /usr/local/bin/swig  (or /opt/homebrew/bin/swig)
swig -version      # should print a version (e.g., 4.2.x)
If you don’t have a compiler, also run once: xcode-select --install

3) # Reinstall Box2D extras for Gym (quote the extras in zsh)
pip install -U pip setuptools wheel
pip install --no-cache-dir "gymnasium[box2d]"
# If it still tries to use the wrong swig, force it:
SWIG="$(which swig)" pip install --no-cache-dir "box2d-py==2.3.5"

4) # Quick sanity test
python - <<'PY'
import gymnasium as gym
env = gym.make("CarRacing-v3", render_mode=None, continuous=False)
obs, info = env.reset()
print("OK:", type(obs), env.observation_space, env.action_space)
env.close()
PY



