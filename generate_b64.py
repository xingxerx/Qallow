import base64, zlib, textwrap
s = open('.github/workflows/internal-ci.yml', 'rb').read()
b = base64.b64encode(zlib.compress(s)).decode('ascii')
print("'''")
for i in range(0, len(b), 80):
    print(b[i:i+80])
print("'''")