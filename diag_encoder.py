import traceback
try:
    from src.model.encoder import DualPathEncoder
    print("OK: DualPathEncoder imported")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()
