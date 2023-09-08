import json

print("Applying Source Files Paches")
with open("/app/patches.json",'r') as pf:
    pjson=json.loads(pf.read())
    for f in pjson:
        fileToPatch=f['name']
        content=""
        with open(fileToPatch,'r') as fpr:
            content=str(fpr.read())
        if content=="":
            continue
        with open(fileToPatch,'w') as fpw:
            for patch in f['patches']:
                content=content.replace(patch['findString'],patch['replaceString'])
            fpw.write(content)