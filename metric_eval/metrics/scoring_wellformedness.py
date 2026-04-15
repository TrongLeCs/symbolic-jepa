import json
import re
import string
from typing import Dict, Optional
from metrics.fol_parser import FOLParser, parsed_tree

class WellFormednessEvaluator:
    def __init__(self):
        self.fol_parser = FOLParser()
    
    def check_standardlize(self, fol_expression):
        # Phat added: use parser to check for standard
        standard_score = 1

        try:
            tree = self.fol_parser.parse(fol_expression)
        except Exception as e:
            standard_score = 0
            tree = None
        
        # return standard score, if it is well formed, then parsing success
        return standard_score, tree

    def check_free_variables(self, fol_expression):
        quantified_vars = re.findall(r'(FORALL|EXISTS)\s+(\w+)', fol_expression)
        quantified_vars = {var[1] for var in quantified_vars}
        expressions_in_parentheses = re.findall(r'\((.*?)\)', fol_expression)
        expression_inside = ' '.join(expressions_in_parentheses)
        all_vars = re.findall(r'\b[a-z_][a-z0-9_]*\b', expression_inside)
        for var in all_vars:
            if len(var) == 1 and var.islower() and var not in quantified_vars:
                return 0
        return 1

    def check_validity(self, fol_expression):
        invalid_patterns = ["AND AND", "OR OR", "NOT OR", "IMPLIES IMPLIES", "IFF IFF", "AND OR", "OR AND", "NOT AND", "IFF IMPLIES", "IMPLIES IFF"]
        for pattern in invalid_patterns:
            if pattern in fol_expression:
                return 0
        return 1

    def check_special_character(self, fol_expression):
        acceptable_keywords = {'FORALL', 'EXISTS', 'NOT', 'AND', 'XOR', 'OR', 'IMPLIES', 'IFF'}
        acceptable_chars = set(string.ascii_letters + string.digits + '()_,')
        tokens = re.findall(r'\b\w+\b|[^\w\s]', fol_expression)
        for token in tokens:
            if token not in acceptable_keywords and not all(char in acceptable_chars for char in token):
                return 0
        return 1

    def check_parentheses_balance(self, fol_expression):
        stack = []
        for char in fol_expression:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return 0
                stack.pop()
        return 1 if len(stack) == 0 else 0

    def check_comparision(self, fol_expression):
        comparisions = ['=', '>', '<']
        for comp in comparisions:
            if comp in fol_expression:
                return 0
        return 1

    def evaluate(self, fol_expression):
        standard_score, tree = self.check_standardlize(fol_expression)
        variable_score = self.check_free_variables(fol_expression)
        validity_score = self.check_validity(fol_expression)
        special_character_score = self.check_special_character(fol_expression)
        parentheses_score = self.check_parentheses_balance(fol_expression)
        comparision_score = self.check_comparision(fol_expression)

        individual_scores = {
            "standard_score": standard_score,
            "variable_score": variable_score,
            "validity_score": validity_score,
            "special_character_score": special_character_score,
            "parentheses_score": parentheses_score,
            "comparision_score": comparision_score
        }

        # parse tree
        parse_tree = parsed_tree(tree)

        score = sum(individual_scores.values()) / len(individual_scores)

        return {
            #"fol_expression": fol_expression,
            "swf_individual_scores": individual_scores,
            "swf_final_score": score,
            "parsed_tree": parse_tree
        }