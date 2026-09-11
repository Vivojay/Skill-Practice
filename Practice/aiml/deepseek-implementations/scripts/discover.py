"""Read-only public-source audit. Downloads only text, never model weights.

Run manually when refreshing the catalogue. Raw sources stay in .research.
Requires network; the actual labs are offline.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "deepseek-minimal-research/0.1"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


def main():
    root = Path('.research')
    root.mkdir(exist_ok=True)
    ids = '2310.16818 2401.02954 2401.06066 2401.14196 2402.03300 2403.05525 2405.04434 2405.14333 2406.11931 2407.01906 2408.08152 2408.14158 2408.15664 2410.13848 2411.07975 2412.10302 2412.19437 2501.12948 2501.17811 2502.11089 2504.02495 2504.21801 2505.09343 2510.18234 2511.22570 2512.02556 2512.24880 2601.07372 2601.20552 2606.19348 2607.05147'.split()
    urls = {i: 'https://arxiv.org/abs/' + i for i in ids}
    for i in ['2401.06066v1', '2405.04434v5', '2412.19437v2', '2402.03300v1', '2408.15664v1', '2605.15403v1']:
        urls[i + '-full'] = 'https://arxiv.org/html/' + i
    urls['github-repos'] = 'https://api.github.com/orgs/deepseek-ai/repos?per_page=100&type=public'
    urls['hf-models'] = 'https://huggingface.co/api/models?author=deepseek-ai&limit=1000'
    urls['news'] = 'https://www.deepseek.com/news/'
    for repo in ['DeepSeek-V3', 'DeepSeek-V2', 'DeepSeek-MoE', 'DeepSeek-Math', 'open-infra-index']:
        urls[repo + '-commit'] = 'https://api.github.com/repos/deepseek-ai/' + repo + '/commits/HEAD'
    report = []
    def one(item):
        key, url = item
        try:
            data = fetch(url)
            (root / (key + '.txt')).write_bytes(data)
            return dict(key=key, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), status='read')
        except Exception as exc:
            return dict(key=key, url=url, status='blocked', error=str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one, urls.items()):
            report.append(row)
            print(row['key'], row['status'], flush=True)
    (root / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    repos = json.loads((root / 'github-repos.txt').read_text(encoding='utf-8'))
    print('REPOSITORIES', [(r['name'], r['default_branch']) for r in repos])
    for repo in ['DeepSeek-V3', 'DeepSeek-V2', 'DeepSeek-MoE', 'DeepSeek-Math', 'open-infra-index']:
        info = json.loads((root / (repo + '-commit.txt')).read_text(encoding='utf-8'))
        sha = info['sha']
        print(repo, sha)
        paths = ['README.md']
        if repo == 'DeepSeek-V3':
            paths += ['inference/model.py', 'LICENSE-CODE']
        for path in paths:
            try:
                (root / (repo + '-' + path.replace('/', '_'))).write_bytes(fetch(f'https://raw.githubusercontent.com/deepseek-ai/{repo}/{sha}/{path}'))
            except Exception as exc:
                print(repo, path, str(exc))


if __name__ == '__main__':
    main()
