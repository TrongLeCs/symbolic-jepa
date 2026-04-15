from metrics.fol_parser import FOLParser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from itertools import product
import numpy as np
import re
import numpy as np
import torch
import time
import random
from typing import List, Tuple

# ---------------- Logic Tools ----------------
class LogicTool:
    def __init__(self):
        self.operators = ['NOT', 'AND', 'XOR', 'OR', 'IMPLIES', 'IFF']
        self.parser = FOLParser()

    def logic_prec(self, op): 
        return {"NOT": 5, "AND": 4, "XOR": 3, "OR": 2, "IMPLIES": 1, "IFF": 0}.get(op, -1)

    def perform_logic(self, op, a, b=None):
        return {
            'NOT': not a,
            'AND': a and b,
            'OR': a or b,
            'XOR': a != b,
            'IMPLIES': not a or b,
            'IFF': a == b
        }.get(op, False)

    def extract_predicates_from_node(self, node, predicates=None):
        if predicates is None:
            predicates = []
        
        if node is None:
            return predicates
        
        if node.type == 'predicate' and '(' in node.value:
            predicates.append(node.value)
        
        self.extract_predicates_from_node(node.left, predicates)
        self.extract_predicates_from_node(node.right, predicates)
        
        return predicates

    def extract_predicate_names_from_list(self, predicates):
        return [pred.split('(')[0] for pred in predicates]

    def extract_variables_from_node(self, node, variables=None):
        if variables is None:
            variables = set()
        
        if node is None:
            return variables
        
        if node.type == 'predicate' and '(' in node.value:
            # Extract args from predicate
            args_part = node.value.split('(')[1].rstrip(')')
            if args_part:
                for arg in args_part.split(','):
                    if arg.strip().isalpha() and len(arg.strip()) == 1:  # Single variable
                        variables.add(arg.strip())
        
        self.extract_variables_from_node(node.left, variables)
        self.extract_variables_from_node(node.right, variables)
        
        return variables

    def evaluate_node(self, node, variable_values):
        if node is None:
            return False
        
        if node.type == 'predicate':
            return variable_values.get(node.value, False)
        
        if node.type == 'operator':
            if node.value == 'NOT':
                return self.perform_logic('NOT', self.evaluate_node(node.left, variable_values))
            else:
                return self.perform_logic(
                    node.value,
                    self.evaluate_node(node.left, variable_values),
                    self.evaluate_node(node.right, variable_values)
                )
                
        return False

# ---------------- Predicate Matcher ----------------
class PredicateMatcher:
    def __init__(self, model):
        self.model = model
        self.logic_tool = LogicTool()

    def extract_predicates(self, expression: str) -> List[str]:
        try:
            root = self.logic_tool.parser.parse(expression)
            return list(set(self.logic_tool.extract_predicates_from_node(root)))
        except Exception:
            # Fallback to regex if parsing fails
            return list(set(re.findall(r'\b\w+\([^)]*\)', expression)))

    def extract_pred_names(self, predicates: List[str]) -> List[str]:
        return [pred.split('(')[0] for pred in predicates]

    def pair_predicates(self, fol: str, fol_pred: str) -> Tuple[List[Tuple[str, str, float]], List[str], List[str]]:
        preds_a = self.extract_predicates(fol)
        preds_b = self.extract_predicates(fol_pred)
        if not preds_a or not preds_b:
            return [], [], []

        # Phat fix: replace _ with ' '
        names_a = [p.replace('_', ' ') for p in self.extract_pred_names(preds_a)]
        names_b = [p.replace('_', ' ') for p in self.extract_pred_names(preds_b)]

        emb_a = self.model.encode(names_a, convert_to_tensor=True)
        emb_b = self.model.encode(names_b, convert_to_tensor=True)
        sim_matrix = cosine_similarity(emb_a.cpu(), emb_b.cpu())

        paired, marked_a, marked_b = [], [], []
        for i in range(len(preds_a)):
            j = int(np.argmax(sim_matrix[i]))
            if np.argmax(sim_matrix[:, j]) == i:
                paired.append((preds_a[i], preds_b[j], float(sim_matrix[i][j])))
                marked_a.append(preds_a[i])
                marked_b.append(preds_b[j])

        unaligned = [(p, '', 0.0) for p in preds_a if p not in marked_a] + [('', p, 0.0) for p in preds_b if p not in marked_b]
        all_paired = paired + unaligned

        torch.cuda.empty_cache()
        return all_paired, [u[0] for u in all_paired], [u[1] if u[1] else u[0] for u in all_paired]

# ---------------- Logic Evaluator ----------------
class LogicEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.matcher = PredicateMatcher(self.model)

    # ---------------- Logic Score Core ----------------
    def _evaluate_logic_score(self, fol_a: str, fol_b: str, var_map_a: List[str], var_map_b: List[str]) -> float:
        # Improvement #1: Limit the number of variables in the truth table
        MAX_VARIABLES = 8
        if len(var_map_a) > MAX_VARIABLES:
            return 0.0
        
        logic_tool = LogicTool()
        
        try:
            # Parse expressions to build AST instead of regex
            node_a = logic_tool.parser.parse(fol_a)
            node_b = logic_tool.parser.parse(fol_b)
            
            # Improvement #3: Optimize truth table generation
            if len(var_map_a) > 6:
                # Take random samples when there are many variables
                sample_size = min(64, 2**len(var_map_a))
                truth_samples = random.sample(list(product([False, True], repeat=len(var_map_a))), sample_size)
            else:
                truth_samples = list(product([False, True], repeat=len(var_map_a)))
            
            match_count = 0
            for values in truth_samples:
                val_map_a = dict(zip(var_map_a, values))
                val_map_b = dict(zip(var_map_b, values))
                
                ra = logic_tool.evaluate_node(node_a, val_map_a)
                rb = logic_tool.evaluate_node(node_b, val_map_b)
                match_count += (ra == rb)
            
            return match_count / len(truth_samples) if truth_samples else 0.0
        
        except Exception as e:
            print(f"Error evaluating logic score: {str(e)}")
            return 0.0

    def evaluate(self, fol: str, fol_pred: str):
        paired, aligned_gold, aligned_pred = self.matcher.pair_predicates(fol, fol_pred)

        try:
            # Add validation for inputs to catch more errors
            if not fol or not fol_pred:
                return 0.0
                
            # Improvement #5: Add checking and exception handling
            if not aligned_gold or not aligned_pred:
                return 0.0
                
            # Try to parse formulas to catch syntax errors
            logic_tool = LogicTool()
            try:
                logic_tool.parser.parse(fol)
                logic_tool.parser.parse(fol_pred)
            except ValueError as e:
                return 0.0
                
            # Process FOL with parser instead of regex
            score = self._evaluate_logic_score(fol, fol_pred, aligned_gold, aligned_pred)
            return score
        except Exception as e:
            # Make sure to catch and return any exceptions that occur
            return 0.0
