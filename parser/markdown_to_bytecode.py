import re


def parse_markdown(md_content):
    header_re = re.compile(r"^#\s+(.*)")
    list_re = re.compile(r"^[-*]\s+(.*)")
    code_block_re = re.compile(r"^```")

    in_code_block = False
    bytecode = []

    for line in md_content.splitlines():
        if code_block_re.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            bytecode.append(compile_code_block(line))
        elif header_re.match(line):
            bytecode.append(handle_header(header_re.match(line).group(1)))
        elif list_re.match(line):
            bytecode.append(handle_list_item(list_re.match(line).group(1)))

    return bytecode


def compile_code_block(code):
    return f"bytecode_for({code.strip()})"


def handle_header(header):
    return f"header({header.strip()})"


def handle_list_item(item):
    return f"list_item({item.strip()})"


if __name__ == "__main__":
    with open("docs/specs/002-organize-codebase/spec.md", "r") as f:
        md = f.read()
    output = parse_markdown(md)
    for line in output:
        print(line)
