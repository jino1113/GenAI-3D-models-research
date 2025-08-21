import csv, glob, os
with open('captions.csv','w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image','text'])
    for p in sorted(glob.glob('*.jpg')):
        w.writerow([os.path.basename(p), 'stainless steel screw on white background'])
