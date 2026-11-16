import os, rasterio

base = r"C:\Users\Saikiran.v\OneDrive - Praval\Desktop\Analysis\SL Analysis\AOI-6\AOI-6 Prospectivity"
print("base exists:", os.path.exists(base))
print("base islink:", os.path.islink(base))
print("base isdir:", os.path.isdir(base))
print("listdir:", os.listdir(base))

rows = []
for root, dirs, files in os.walk(base, followlinks=True):
    for f in files:
        if f.lower().endswith((".tif", ".tiff")):
            p = os.path.join(root, f)
            try:
                with rasterio.open(p) as s:
                    rows.append((str(s.crs), s.transform.a, s.width, s.height, p))
            except Exception as e:
                rows.append(("ERR", str(e), 0, 0, p))

for r in rows:
    print(f"{str(r[0]):20s} | {r[1]:10.4f} | {r[2]:5d}x{r[3]:<5d} | {r[4]}")