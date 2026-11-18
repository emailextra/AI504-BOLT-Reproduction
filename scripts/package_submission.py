"""Create the instructor's shared-folder layout and Blackboard filenames."""
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package():
    """Copy final artifacts using the allocated group number when available."""
    details = json.loads((ROOT / 'submission-details.json').read_text())
    number = str(details.get('group_number') or 'NN')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', number):
        raise ValueError('Group number must contain letters, digits, underscores or hyphens')
    stem = 'AI504-A2-Group' + number
    folder = ROOT / 'submission' / stem
    papers = folder / 'Papers'
    blackboard = ROOT / 'submission' / 'Blackboard'
    papers.mkdir(parents=True, exist_ok=True)
    blackboard.mkdir(parents=True, exist_ok=True)
    for name in ('Report.pdf', 'Slides.pdf', 'Repository-link.txt'):
        shutil.copy2(ROOT / name, folder / name)
    for source in (ROOT / 'Papers').glob('*.pdf'):
        shutil.copy2(source, papers / source.name)
    for artifact in ('Report.pdf', 'Slides.pdf', 'Slides.pptx'):
        shutil.copy2(ROOT / artifact, blackboard / f'{stem}-{artifact}')
    archive = ROOT / 'submission' / f'{stem}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                bundle.write(source, Path(stem) / source.relative_to(folder))
    print('Shared-folder contents:', folder)
    print('Blackboard copies:', blackboard)
    if number == 'NN':
        print('NN is the group-number placeholder approved for this preparation stage.')


if __name__ == '__main__':
    package()
