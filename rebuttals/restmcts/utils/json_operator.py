import json
import os.path


def read_json(source):
    json_list = []
    if not os.path.exists(source):
        return json_list
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, item in enumerate(datas):
            json.dump(item, f, ensure_ascii=False, indent=4)
            if i < len(datas) - 1:  # Only add a comma if it's not the last item
                f.write(',\n')
        f.write('\n]')
