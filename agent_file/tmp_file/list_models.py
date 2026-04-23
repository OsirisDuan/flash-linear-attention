import csv
from collections import defaultdict

models = defaultdict(lambda: {'heat': '', 'operators': set(), 'layers': set()})

with open(r'd:\code\flash-linear-attention\agent_file\tmp_file\t1_model_operator_mapping.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        name = r['模型名称']
        models[name]['heat'] = r['模型热度级别']
        if r['算子名称']:
            models[name]['operators'].add(r['算子名称'])
        if r['Layer名称']:
            models[name]['layers'].add(r['Layer名称'])

print('模型列表:')
heat_order = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
for name in sorted(models.keys(), key=lambda x: (heat_order.get(models[x]['heat'], 5), x)):
    m = models[name]
    ops = ', '.join(sorted(m['operators']))
    print(f'{name} ({m["heat"]}): 算子({len(m["operators"])})=[{ops}]')

print(f'\n总计: {len(models)} 个模型')
