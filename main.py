import argparse
from pathlib import Path
from typing import Optional

from cc_net import dedup, jsonql, minify, perplexity, split_by_lang

class ManualMinifier(jsonql.Transformer):
    """Manually minifies documents, keeping only specific fields."""

    ready = True  # No preparation needed

    def __init__(self):
        self.fields_to_keep = {
            "url",
            "raw_content",
            "digest",
            "source_domain",
            "title",
            "date_download",
            "bucket",
            "language_score",
            "length",
            "nlines",
            "original_length",
            "original_nlines",
            "perplexity",
            "language",
        }

    def do(self, doc: dict) -> Optional[dict]:
        """Keeps only the specified fields in the document."""
        if not doc:
            return None
        # Create a new dictionary with only the desired fields
        minified_doc = {k: v for k, v in doc.items() if k in self.fields_to_keep}
        return minified_doc


def compute_hashes(input_file: Path, output_file: Path):
    """Computes hashes for all documents and saves them to a file."""
    hash_collector = dedup.HashesCollector(field="raw_content", output=output_file)
    jsonql.run_pipes(hash_collector, file=input_file)


def main(
    file: Path,
    output: Path,
    min_len: int,
    lang_id_model: Path,
    lm_dir: Path,
    cutoff_csv: Path,
):
    # Compute hashes and store them in a file
    hashes_file = args.file.with_suffix(".hashes")
    compute_hashes(args.file, hashes_file)

    # Define processing pipeline using precomputed hashes
    pipeline = [
        jsonql.JsonReader(),  # Read JSON documents
        perplexity.RemoveSmall(field="raw_content", min_len=args.min_len),  # Filter by minimum length
        dedup.DuplicatesRemover(
            field="raw_content", hashes_files=[hashes_file]
        ),  # Deduplication
        split_by_lang.Classifier(  # Language identification
            model=args.lang_id_model,
            field="raw_content",
            out_field="language",
            top=1,
        ),
        jsonql.where([lambda doc: doc.get("language") == "tr"]),  # Keep only Turkish
        perplexity.SentencePiece(  # Tokenization
            model=args.lm_dir / "tr.sp.model",
            field="raw_content",
            output_field="tokenized",
            normalize=True,
        ),
        perplexity.DocLM(  # Perplexity calculation
            models=args.lm_dir / "tr.arpa.bin",
            field="tokenized",
            output_field="perplexity",
            normalize=False,
        ),
        perplexity.PerplexityBucket(cutoff_csv=args.cutoff_csv),  # Perplexity bucketing
        ManualMinifier(),  # Minify fields
    ]

    jsonql.run_pipes(*pipeline, file=args.file, output=args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Common Crawl data")
    parser.add_argument("--file", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument(
        "--min_len", type=int, default=300, help="Minimum document length to keep"
    )
    parser.add_argument(
        "--lang_id_model",
        type=Path,
        required=True,
        help="Path to the language identification model",
    )
    parser.add_argument(
        "--lm_dir",
        type=Path,
        required=True,
        help="Path to the directory containing language models",
    )
    parser.add_argument(
        "--cutoff_csv",
        type=Path,
        required=True,
        help="Path to the CSV file with perplexity cutoffs",
    )
    args = parser.parse_args()

    main(**vars(args))  # Pass the dictionary to main
