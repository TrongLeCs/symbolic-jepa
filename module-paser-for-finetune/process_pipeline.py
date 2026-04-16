import argparse
import json
import logging
import time
from pathlib import Path

from transformers import T5TokenizerFast

from parser.ast_paths_builder import ASTPathsDatasetBuilderASTOnly
from parser.dfg_links_builder import DFGLinksDatasetBuilderT5
from parser.fol_parser_builder import parse_and_dump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("process_pipeline")

def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline for processing Logic-JEPA DFG, AST, and Text trees"
    )
    parser.add_argument("--input", required=True, help="Path to input JSON (list of records).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated outputs.")
    parser.add_argument("--prefix", default="samples", help="Prefix for output files (e.g., 'val' -> val_ast_paths.npz).")
    parser.add_argument("--tokenizer", default="t5-base", help="T5 Tokenizer version to use.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum AST depth.")
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
    
    # 1. AST Paths Generation
    logger.info("Initializing AST Builder...")
    ast_output_npz = output_dir / f"{args.prefix}_ast_paths.npz"
    ast_error_log = output_dir / f"{args.prefix}_ast_errors.jsonl"
    
    ast_builder = ASTPathsDatasetBuilderASTOnly(
        tokenizer=tokenizer,
        max_depth=args.max_depth,
        max_length=args.max_length
    )
    
    logger.info("Building AST Paths...")
    ast_stats = ast_builder.build_and_save(input_path, ast_output_npz, ast_error_log)
    logger.info(f"AST Paths Done: {ast_stats['num_ok']}/{ast_stats['num_records']} successful.")
    
    # 2. DFG Links Generation
    logger.info("Initializing DFG Links Builder...")
    dfg_output_npz = output_dir / f"{args.prefix}_dfg_links.npz"
    dfg_error_log = output_dir / f"{args.prefix}_dfg_errors.jsonl"
    
    dfg_builder = DFGLinksDatasetBuilderT5(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    logger.info("Building DFG Links...")
    dfg_stats = dfg_builder.build_and_save(input_path, dfg_output_npz, dfg_error_log)
    logger.info(f"DFG Links Done: {dfg_stats['num_ok']}/{dfg_stats['num_records']} successful.")
    
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
