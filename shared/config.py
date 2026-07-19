import os

class Settings:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self, env_path: str = None):
        if self._initialized:
            return
        
        if env_path is None:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        
        self.env_path = env_path
        self.settings_dict = {}
        self.load_env()
        self._initialized = True

    def load_env(self):
        if os.path.exists(self.env_path):
            try:
                with open(self.env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, val = line.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            self.settings_dict[key] = val
            except Exception as e:
                print(f"Error loading .env file: {e}")

        for key, val in os.environ.items():
            self.settings_dict[key] = val

    def get(self, key: str, default=None):
        return self.settings_dict.get(key, default)

    def __getattr__(self, name):
        if name in self.settings_dict:
            return self.settings_dict[name]
        return None

_settings_instance = None

def get_settings(env_path: str = None) -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings(env_path)
    return _settings_instance
