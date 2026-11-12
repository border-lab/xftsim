"""Build stdpopsim founders once and cache them (GRG + metadata) for the edge experiments."""
import argparse, os, pickle, time, warnings
warnings.filterwarnings("ignore")
import pygrgl
import xftsim.founders as xf

ap = argparse.ArgumentParser()
ap.add_argument("--people", type=int, required=True)
ap.add_argument("--chrom", default="chr22")
ap.add_argument("--left-mb", type=float, default=17.0)
ap.add_argument("--length-mb", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()

t = time.perf_counter()
left = int(a.left_mb * 1e6)
f = xf.founder_haplotypes_from_stdpopsim_grg(
    samples={"CEU": a.people}, model_id="OutOfAfrica_3G09", chromosome=a.chrom,
    genetic_map="HapMapII_GRCh38", left=left, right=left + int(a.length_mb * 1e6), seed=a.seed)
print(f"founders {time.perf_counter()-t:.1f}s n={f.n} m={f.m} nodes={f._grg.num_nodes} edges={f._grg.num_edges}")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pygrgl.save_grg(f._grg, a.out + ".grg", allow_simplify=False)
with open(a.out + ".meta.pkl", "wb") as fh:
    pickle.dump({"samples": f.samples, "variants": f.variants}, fh)
g2 = pygrgl.load_mutable_grg(a.out + ".grg")
print("reloaded", g2.num_nodes, g2.num_edges, g2.num_samples, g2.num_mutations)
