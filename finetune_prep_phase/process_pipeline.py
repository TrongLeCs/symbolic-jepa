import argparse
import json
import logging
import time
from pathlib import Path

from transformers import T5TokenizerFast

from parser.cpp_paths_builder import CPPPathsDatasetBuilderCPPOnly
from parser.ldp_links_builder import LDPLinksDatasetBuilderT5
from parser.fol_parser_builder import parse_and_dump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("process_pipeline")

def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline for processing Logic-JEPA LDP, CPP, and Text trees"
    )
    parser.add_argument("--input", required=True, help="Path to input JSON (list of records).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated outputs.")
    parser.add_argument("--prefix", default="samples", help="Prefix for output files (e.g., 'val' -> val_cpp_paths.npz).")
    parser.add_argument("--tokenizer", default="t5-base", help="T5 Tokenizer version to use.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum CPP depth.")
    parser.add_argument("--dump_text_tree", action="store_true", help="Dump human-readable text tree representation.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    start_time = time.time()
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer)
    logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
    
    # 1. CPP Paths Generation
    logger.info("Initializing CPP Builder...")
    cpp_output_npz = output_dir / f"{args.prefix}_cpp_paths.npz"
    cpp_error_log = output_dir / f"{args.prefix}_cpp_errors.jsonl"
    
    cpp_builder = CPPPathsDatasetBuilderCPPOnly(
        tokenizer=tokenizer,
        max_depth=args.max_depth,
        max_length=args.max_length
    )
    
    logger.info("Building CPP Paths...")
    cpp_stats = cpp_builder.build_and_save(input_path, cpp_output_npz, cpp_error_log)
    logger.info(f"CPP Paths Done: {cpp_stats['num_ok']}/{cpp_stats['num_records']} successful.")
    
    # 2. LDP Links Generation
    logger.info("Initializing LDP Links Builder...")
    ldp_output_npz = output_dir / f"{args.prefix}_ldp_links.npz"
    ldp_error_log = output_dir / f"{args.prefix}_ldp_errors.jsonl"
    
    ldp_builder = LDPLinksDatasetBuilderT5(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    logger.info("Building LDP Links...")
    ldp_stats = ldp_builder.build_and_save(input_path, ldp_output_npz, ldp_error_log)
    logger.info(f"LDP Links Done: {ldp_stats['num_ok']}/{ldp_stats['num_records']} successful.")
    
    # 3. Text Tree Dump (Optional)
    if args.dump_text_tree:
        logger.info("Dumping Text Tree Representation...")
        tree_output_txt = output_dir / f"{args.prefix}_fol_trees.txt"
        tree_error_log = output_dir / f"{args.prefix}_tree_errors.jsonl"
        tree_stats = parse_and_dump(input_path, tree_output_txt, tree_error_log)
        logger.info(f"Text Tree Dump Done: {tree_stats['num_ok']}/{tree_stats['num_records']} successful.")

    logger.info(f"All processing complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
