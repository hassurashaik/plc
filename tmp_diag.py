import traceback
try:
    from model.decoder import DualPathEncoder
    print("OK: DualPathEncoder imported")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()
