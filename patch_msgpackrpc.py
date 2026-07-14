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

    original_content = content

    # Apply fixes for Python 3 and modern msgpack.
    if "e.message" in content:
        content = content.replace("e.message", "e.args[0]")

    content = content.replace(
        "msgpack.Packer(encoding=encodings[0], default=lambda x: x.to_msgpack())",
        "msgpack.Packer(default=lambda x: x.to_msgpack())",
    )
    content = content.replace(
        "msgpack.Unpacker(encoding=encodings[1])",
        "msgpack.Unpacker(raw=False)",
    )

    if content != original_content:
        with open(tcp_py, "w", encoding="utf-8") as f:
            f.write(content)
        print("msgpackrpc successfully patched for Python 3 and modern msgpack!")
    else:
        print("msgpackrpc is already patched.")

if __name__ == "__main__":
    patch_msgpackrpc()
