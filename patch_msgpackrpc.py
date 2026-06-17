import os
import sys

def patch_msgpackrpc():
    try:
        import msgpackrpc
    except ImportError:
        print("msgpack-rpc-python is not installed. Please run `pip install -r foreground/requirements.txt` first.")
        sys.exit(1)

    # find the location of the installed module
    module_path = os.path.dirname(msgpackrpc.__file__)
    tcp_py = os.path.join(module_path, "transport", "tcp.py")

    if not os.path.exists(tcp_py):
        print(f"Could not find tcp.py at {tcp_py}")
        sys.exit(1)

    with open(tcp_py, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply the fix for Python 3+ (Exception.message was removed)
    if "e.message" in content:
        content = content.replace("e.message", "e.args[0]")
        with open(tcp_py, "w", encoding="utf-8") as f:
            f.write(content)
        print("msgpackrpc successfully patched for Python 3 compatibility!")
    else:
        print("msgpackrpc is already patched or 'e.message' not found.")

if __name__ == "__main__":
    patch_msgpackrpc()
