"""Hash pure training-donor candidates before any CSL pose evaluation."""
from __future__ import annotations
import argparse
import json
import pickle
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_csl_text_route_transfer import array, confined_output_path
from scripts.build_retrieval_gloss_plans import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ROOT/"outputs/revision/astra_open_closure_20260906/csl_whole_candidates"))
    parser.add_argument("--v4_dir", default=str(ROOT/"outputs/revision/astra_open_closure_20260906/csl_text_transfer_v4"))
    args = parser.parse_args()
    out, v4 = confined_output_path(args.out_dir), confined_output_path(args.v4_dir)
    if out.exists():
        raise ValueError("use a new directory for immutable whole-candidate banks")
    protocol = json.loads((v4/"protocol.json").read_text())
    if protocol["version"] != "CSL-text-native-fixed40-v4":
        raise ValueError("expected frozen v4 construction contract")
    archive_path = Path(protocol["inputs"]["pose_archive"]["path"])
    train_path = Path(protocol["inputs"]["train_manifest"]["path"])
    for name, path in (("pose_archive", archive_path), ("train_manifest", train_path)):
        if sha256(path) != protocol["inputs"][name]["sha256"]:
            raise ValueError(f"v4 input hash mismatch: {name}")
    allowed = {row["id"] for row in json.loads(train_path.read_text())}
    with archive_path.open("rb") as handle:
        archive = pickle.load(handle)
    out.mkdir(parents=True)
    provenance = {"query_fields_read": ["id", "retrieved_id", "exact_caption_source_ids"],
                  "source": "admitted raw training archive; unchanged float32 native tensors",
                  "no_query_annotations_or_poses": True, "protocol_sha256": sha256(v4/"protocol.json"),
                  "admission_sha256": sha256(v4/"archive_admission.json"),
                  "archive_sha256": sha256(archive_path), "train_manifest_sha256": sha256(train_path),
                  "implementation_sha256": sha256(Path(__file__)), "splits": {}}
    for split in ("dev", "test"):
        trace = json.loads((v4/f"{split}_trace.json").read_text())
        prescore = json.loads((v4/f"{split}_prescore_hashes.json").read_text())
        if prescore[f"{split}_trace.json"] != sha256(v4/f"{split}_trace.json"):
            raise ValueError("v4 trace prescore hash mismatch")
        poses, donor_ids = {}, set()
        for row in trace:
            sid, donor = row["id"], row["retrieved_id"]
            if sid in allowed or sid in poses or donor not in allowed or donor not in archive:
                raise ValueError("candidate source/identifier partition violation")
            if donor in row["exact_caption_source_ids"]:
                raise ValueError("exact-caption donor cannot be replayed")
            poses[sid] = torch.from_numpy(array(archive[donor]))
            donor_ids.add(donor)
        path = out/f"{split}_whole.pt"
        torch.save(poses, path)
        provenance["splits"][split] = {"n": len(poses), "frames": sum(len(v) for v in poses.values()),
                                        "donors": len(donor_ids), "bank": str(path), "sha256": sha256(path),
                                        "trace_sha256": sha256(v4/f"{split}_trace.json")}
        print(json.dumps(provenance["splits"][split]), flush=True)
    provenance["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
