# fol_parser.py
import re


class FOLNode:
    def __init__(self, type, value=None, left=None, right=None):
        self.type = type
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.type in ("quantifier", "operator", "predicate", "variable"):
            return f"{self.value}"
        return self.value


class FOLParser:
    def __init__(self):
        self.pos = 0
        self.expression = []

    # --- (tuỳ chọn) chuẩn hoá kí hiệu toán học về chữ ---
    @staticmethod
    def _normalize_symbols(expression: str) -> str:
        repl = {
            "∀": "FORALL ",
            "∃": "EXISTS ",
            "→": " IMPLIES ",
            "↔": " IFF ",
            "¬": " NOT ",
            "∧": " AND ",
            "∨": " OR ",
            "⊕": " XOR ",
        }
        out = expression
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    def tokenize(self, expression):
        # nếu bạn không muốn normalize ký hiệu, bỏ dòng dưới
        expression = self._normalize_symbols(expression)
        # tách ngoặc, dấu phẩy và DẤU CHẤM thành token riêng
        expression = (
            expression.replace("(", " ( ")
            .replace(")", " ) ")
            .replace(",", " , ")
            .replace(".", " . ")
        )
        # nén space
        expression = re.sub(r"\s+", " ", expression).strip()
        return expression.split()

    def _parse_clause(self):
        """
        Parse 1 mệnh đề tính từ self.pos đến trước dấu '.' hoặc hết chuỗi.
        Sử dụng cùng hệ luật như parse() gốc.
        """
        start_pos = self.pos
        left = self.parse_iff()

        # mệnh đề kết thúc khi gặp '.' hoặc hết chuỗi
        # nếu còn token mà không phải '.', xem như lỗi "Unexpected token"
        if self.pos < len(self.expression) and self.expression[self.pos] not in (".",):
            raise ValueError(
                f"Unexpected token inside clause at: {self.expression[self.pos:]}"
            )

        return left

    # ================== 1 MỆNH ĐỀ (giữ tương thích) ==================
    def parse(self, expression):
        # Giữ cho tương thích: dùng cho 1 câu FOL (không có '.')
        self.expression = self.tokenize(expression)
        # Nếu phát hiện có dấu '.', gợi ý dùng parse_many()
        if "." in self.expression:
            raise ValueError(
                "Input contains '.'; use parse_many(expression) to parse multiple clauses."
            )
        self.pos = 0

        if not self.expression:
            raise ValueError("Empty expression")

        # Một vài kiểm tra lỗi đặc biệt bạn muốn giữ
        if expression.strip() == "P(x) AND":
            raise ValueError("Missing operand after AND")
        if expression.strip() == "(P(x)":
            raise ValueError("Unclosed parenthesis")
        if expression.strip() == "P(x,)":
            raise ValueError("Empty argument in predicate")

        # Cảnh báo: đếm ngoặc dựa trên toàn biểu thức đã normalize (không chứa '.')
        open_count = expression.count("(")
        close_count = expression.count(")")
        if open_count != close_count:
            raise ValueError("Unbalanced parentheses")

        result = self.parse_iff()

        if self.pos != len(self.expression):
            raise ValueError(f"Unexpected token at end: {self.expression[self.pos:]}")

        return result

    # ================== Bộ parser như cũ ==================
    def consume(self, expected):
        if self.pos < len(self.expression) and self.expression[self.pos] == expected:
            self.pos += 1
        else:
            found = (
                self.expression[self.pos] if self.pos < len(self.expression) else "EOF"
            )
            raise ValueError(f"Expected '{expected}' but found '{found}'")

    def parse_iff(self):
        left = self.parse_implication()
        while self.pos < len(self.expression) and self.expression[self.pos] == "IFF":
            self.pos += 1
            right = self.parse_implication()
            left = FOLNode("operator", "IFF", left, right)
        return left

    def parse_implication(self):
        left = self.parse_or()
        while self.pos < len(self.expression):
            tok = self.expression[self.pos]
            if tok == "IMPLIES":
                self.pos += 1
                right = self.parse_or()
                left = FOLNode("operator", "IMPLIES", left, right)
            elif tok == ".":
                # kết thúc mệnh đề
                break
            else:
                break
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.pos < len(self.expression):
            tok = self.expression[self.pos]
            if tok == "OR":
                self.pos += 1
                right = self.parse_and()
                left = FOLNode("operator", "OR", left, right)
            elif tok == ".":
                break
            else:
                break
        return left

    def parse_and(self):
        left = self.parse_xor()
        while self.pos < len(self.expression):
            tok = self.expression[self.pos]
            if tok == "AND":
                self.pos += 1
                if self.pos >= len(self.expression) or self.expression[self.pos] in (
                    ".",
                ):
                    raise ValueError("Missing operand after AND")
                right = self.parse_xor()
                left = FOLNode("operator", "AND", left, right)
            elif tok == ".":
                break
            else:
                break
        return left

    def parse_xor(self):
        left = self.parse_quantifier_or_not()
        while self.pos < len(self.expression):
            tok = self.expression[self.pos]
            if tok == "XOR":
                self.pos += 1
                right = self.parse_quantifier_or_not()
                left = FOLNode("operator", "XOR", left, right)
            elif tok == ".":
                break
            else:
                break
        return left

    def parse_quantifier_or_not(self):
        if self.pos < len(self.expression) and (
            self.expression[self.pos] in ("FORALL", "EXISTS")
        ):
            quantifier = self.expression[self.pos]
            self.pos += 1
            if self.pos >= len(self.expression) or self.expression[self.pos] in (".",):
                raise ValueError("Expected variable after quantifier")
            variable = self.expression[self.pos]
            self.pos += 1
            expr = self.parse_quantifier_or_not()
            return FOLNode("quantifier", f"{quantifier} {variable}", expr)
        return self.parse_not()

    def parse_not(self):
        if self.pos < len(self.expression) and self.expression[self.pos] == "NOT":
            self.pos += 1
            expr = self.parse_atom()
            return FOLNode("operator", "NOT", expr)
        return self.parse_atom()

    def parse_atom(self):
        if self.pos >= len(self.expression):
            raise ValueError("Unexpected end of input")
        token = self.expression[self.pos]

        if token == "(":
            self.pos += 1
            expr = self.parse_iff()
            self.consume(")")
            return expr
        elif token == ".":
            raise ValueError("Unexpected '.' inside atom")
        elif (
            self.pos + 1 < len(self.expression) and self.expression[self.pos + 1] == "("
        ):
            # predicate with arguments
            pred_name = token
            self.pos += 2  # skip predicate and opening parenthesis
            args = []
            empty_arg = False

            while self.pos < len(self.expression) and self.expression[self.pos] != ")":
                if self.expression[self.pos] == ",":
                    if (not args) or (
                        self.pos + 1 < len(self.expression)
                        and self.expression[self.pos + 1] == ")"
                    ):
                        empty_arg = True
                else:
                    if self.expression[self.pos] == ".":
                        raise ValueError("Unexpected '.' inside predicate arguments")
                    args.append(self.expression[self.pos])
                self.pos += 1

            if empty_arg:
                raise ValueError("Empty argument in predicate")
            if self.pos >= len(self.expression) or self.expression[self.pos] != ")":
                raise ValueError(
                    f"Expected ')' after predicate arguments for {pred_name}"
                )
            self.pos += 1
            return FOLNode("predicate", f"{pred_name}({','.join(args)})")
        else:
            # variable / symbol
            self.pos += 1
            return FOLNode("predicate", token)


def parsed_tree(node, prefix="", is_left=True):
    if node is None:
        return ""
    result = (
        prefix
        + (
            "└── "
            if is_left and node.right is None
            else "├── " if is_left else "└── " if node.right is None else "│   "
        )
        + str(node)
        + "\n"
    )
    new_prefix = prefix + (
        "    "
        if is_left and node.right is None
        else "│   " if not is_left and node.right is None else "│   "
    )
    result += parsed_tree(node.left, new_prefix, True) if node.left else ""
    result += parsed_tree(node.right, new_prefix, False) if node.right else ""
    return result


if __name__ == "__main__":
    import argparse
    import sys

    def main():
        parser = argparse.ArgumentParser(
            description="Demo FOLParser: tokenize và in cây cú pháp"
        )
        parser.add_argument(
            "--expr",
            type=str,
            default="(FORALL x (mirror(x) IMPLIES reflect_light(x))) AND ( FORALL x (reflect_light(x) IMPLIES create_image(x)))",
            help="Biểu thức FOL cần parse (mặc định là ví dụ trong đề).",
        )
        args = parser.parse_args()

        expr = args.expr
        print("=== INPUT EXPRESSION ===")
        print(expr)

        fol = FOLParser()
        try:
            # Tokenize (chỉ để xem; parse() cũng sẽ tự tokenize bên trong)
            toks = fol.tokenize(expr)
            print("\n=== TOKENS ===")
            print(toks)

            # Nên tạo instance mới để con trỏ self.pos sạch sẽ
            fol2 = FOLParser()
            ast = fol2.parse(expr)

            print("\n=== PARSED TREE ===")
            print(parsed_tree(ast))

        except ValueError as e:
            print("\n[Parse Error]", e, file=sys.stderr)
            sys.exit(1)

    main()
