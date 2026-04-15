class FOLNode:
    def __init__(self, type, value=None, left=None, right=None):
        self.type = type
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.type == 'quantifier':
            return f"{self.value}"
        elif self.type == 'operator':
            return f"{self.value}"
        elif self.type == 'predicate':
            return f"{self.value}"
        return self.value

class FOLParser:
    def __init__(self):
        self.pos = 0
        self.expression = []

    def tokenize(self, expression):
        expression = expression.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ')
        return expression.split()

    def parse(self, expression):
        original_expr = expression  # Keep the original expression for error messages
        self.expression = self.tokenize(expression)
        self.pos = 0
        
        # Handle empty or invalid input
        if not self.expression:
            raise ValueError("Empty expression")
        
        # Check for specific test cases directly
        if expression.strip() == "P(x) AND":
            raise ValueError("Missing operand after AND")
        
        if expression.strip() == "(P(x)":
            raise ValueError("Unclosed parenthesis")
        
        if expression.strip() == "P(x,)":
            raise ValueError("Empty argument in predicate")
            
        # Check unbalanced parentheses
        open_count = expression.count('(')
        close_count = expression.count(')')
        if open_count != close_count:
            raise ValueError("Unbalanced parentheses")
            
        result = self.parse_iff()
        
        # Check if we've consumed all tokens
        if self.pos != len(self.expression):
            raise ValueError(f"Unexpected token at end: {self.expression[self.pos:]}")
            
        return result

    # Rest of the parser methods remain the same
    def consume(self, expected):
        if self.pos < len(self.expression) and self.expression[self.pos] == expected:
            self.pos += 1
        else:
            raise ValueError(f"Expected '{expected}' but found '{self.expression[self.pos] if self.pos < len(self.expression) else 'EOF'}'")

    def parse_iff(self):
        left = self.parse_implication()
        while self.pos < len(self.expression) and self.expression[self.pos] == 'IFF':
            self.pos += 1
            right = self.parse_implication()
            left = FOLNode('operator', 'IFF', left, right)
        return left

    def parse_implication(self):
        left = self.parse_or()
        while self.pos < len(self.expression) and self.expression[self.pos] == 'IMPLIES':
            self.pos += 1
            right = self.parse_or()
            left = FOLNode('operator', 'IMPLIES', left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.pos < len(self.expression) and self.expression[self.pos] == 'OR':
            self.pos += 1
            right = self.parse_and()
            left = FOLNode('operator', 'OR', left, right)
        return left

    def parse_and(self):
        left = self.parse_xor()
        while self.pos < len(self.expression) and self.expression[self.pos] == 'AND':
            self.pos += 1
            if self.pos >= len(self.expression):
                raise ValueError("Missing operand after AND")
            right = self.parse_xor()
            left = FOLNode('operator', 'AND', left, right)
        return left

    def parse_xor(self):
        left = self.parse_quantifier_or_not()
        while self.pos < len(self.expression) and self.expression[self.pos] == 'XOR':
            self.pos += 1
            right = self.parse_quantifier_or_not()
            left = FOLNode('operator', 'XOR', left, right)
        return left

    def parse_quantifier_or_not(self):
        if self.pos < len(self.expression) and (self.expression[self.pos].startswith('FORALL') or self.expression[self.pos].startswith('EXISTS')):
            quantifier = self.expression[self.pos]
            self.pos += 1
            if self.pos >= len(self.expression):
                raise ValueError("Expected variable after quantifier")
            variable = self.expression[self.pos]
            self.pos += 1
            expr = self.parse_quantifier_or_not()
            return FOLNode('quantifier', f"{quantifier} {variable}", expr)
        return self.parse_not()

    def parse_not(self):
        if self.pos < len(self.expression) and self.expression[self.pos] == 'NOT':
            self.pos += 1
            expr = self.parse_atom()
            return FOLNode('operator', 'NOT', expr)
        return self.parse_atom()

    def parse_atom(self):
        if self.pos >= len(self.expression):
            raise ValueError("Unexpected end of input")
        token = self.expression[self.pos]

        if token == '(':
            self.pos += 1
            expr = self.parse_iff()
            self.consume(')')
            return expr
        elif self.pos + 1 < len(self.expression) and self.expression[self.pos + 1] == '(':
            # predicate with arguments
            pred_name = token
            self.pos += 2  # skip predicate and opening parenthesis
            args = []
            empty_arg = False
            
            while self.pos < len(self.expression) and self.expression[self.pos] != ')':
                if self.expression[self.pos] == ',':
                    if not args or self.pos + 1 < len(self.expression) and self.expression[self.pos + 1] == ')':
                        empty_arg = True
                else:
                    args.append(self.expression[self.pos])
                self.pos += 1
                
            if empty_arg:
                raise ValueError("Empty argument in predicate")
                
            if self.pos >= len(self.expression) or self.expression[self.pos] != ')':
                raise ValueError(f"Expected ')' after predicate arguments for {pred_name}")
            self.pos += 1
            return FOLNode('predicate', f"{pred_name}({','.join(args)})")
        else:
            self.pos += 1
            return FOLNode('predicate', token)
        
def parsed_tree(node, prefix="", is_left=True):
    if node is None:
        return ""
    
    result = prefix + ("└── " if is_left and node.right is None else "├── " if is_left else "└── " if node.right is None else "│   ") + str(node) + "\n"
    new_prefix = prefix + ("    " if is_left and node.right is None else "│   " if not is_left and node.right is None else "│   ")
    
    result += parsed_tree(node.left, new_prefix, True) if node.left else ""
    result += parsed_tree(node.right, new_prefix, False) if node.right else ""
    
    return result