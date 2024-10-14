// Rust implementation of llama grammar parser directly translated from C++ source file in vendor/llama.cpp/common/grammar-parser.cpp.

use lazy_static::lazy_static;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self};
use std::path::PathBuf;
use std::str::FromStr;
static LLAMA_GRAMMAR_DEFAULT_ROOT: &str = "root";

struct LlamaGrammar {
    _grammar: String,
    _root: String,
}

impl LlamaGrammar {
    pub fn new(_grammar: impl AsRef<str>) -> Self {
        Self {
            _grammar: _grammar.as_ref().to_string(),
            _root: LLAMA_GRAMMAR_DEFAULT_ROOT.to_string(),
        }
    }

    fn from_string(_grammar: impl AsRef<str>) -> Self {
        Self::new(_grammar)
    }

    fn from_file(_grammar_file: impl AsRef<PathBuf>) -> io::Result<Self> {
        let grammar = std::fs::read_to_string(_grammar_file.as_ref().as_path())?;
        if grammar.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Grammar file is empty",
            ));
        }
        Ok(Self::new(grammar))
    }

    fn from_json_schema(schema: impl AsRef<str>) -> Self {
        let grammar = json_schema_to_gbnf(schema, None);
        Self::new(grammar)
    }
}

// llama.cpp gbnf rules from vendor/llama.cpp/grammars

const ARITHMETIC_GBNF: &str = r#"
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"#;

const C_GBNF: &str = r#"
root ::= (declaration)*

declaration ::= dataType identifier "(" parameter? ")" "{" statement* "}"

dataType  ::= "int" ws | "float" ws | "char" ws
identifier ::= [a-zA-Z_] [a-zA-Z_0-9]*

parameter ::= dataType identifier

statement ::=
    ( dataType identifier ws "=" ws expression ";" ) |
    ( identifier ws "=" ws expression ";" ) |
    ( identifier ws "(" argList? ")" ";" ) |
    ( "return" ws expression ";" ) |
    ( "while" "(" condition ")" "{" statement* "}" ) |
    ( "for" "(" forInit ";" ws condition ";" ws forUpdate ")" "{" statement* "}" ) |
    ( "if" "(" condition ")" "{" statement* "}" ("else" "{" statement* "}")? ) |
    ( singleLineComment ) |
    ( multiLineComment )

forInit ::= dataType identifier ws "=" ws expression | identifier ws "=" ws expression
forUpdate ::= identifier ws "=" ws expression

condition ::= expression relationOperator expression
relationOperator ::= ("<=" | "<" | "==" | "!=" | ">=" | ">")

expression ::= term (("+" | "-") term)*
term ::= factor(("*" | "/") factor)*

factor ::= identifier | number | unaryTerm | funcCall | parenExpression
unaryTerm ::= "-" factor
funcCall ::= identifier "(" argList? ")"
parenExpression ::= "(" ws expression ws ")"

argList ::= expression ("," ws expression)*

number ::= [0-9]+

singleLineComment ::= "//" [^\n]* "\n"
multiLineComment ::= "/*" ( [^*] | ("*" [^/]) )* "*/"

ws ::= ([ \t\n]+)
"#;

const CHESS_GBNF: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"#;

const JAPANESE_GBNF: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"#;

const JSON_ARR_GBNF: &str = r#"
# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays

root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
"#;

const JSON_GBNF: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"#;

const LIST_GBNF: &str = r#"
root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"
"#;

// llama.cpp json-schema to grammar converter from vendor/llama.cpp/examples/json-schema-to-grammar.py

// whitespace is constrained to a single space char to prevent model "running away" in
// whitespace. Also maybe improves generation quality?
const SPACE_RULE: &str = "\" \"?";

lazy_static! {
    static ref INVALID_RULE_CHARS_RE: Regex = Regex::new(r"[^a-zA-Z0-9-]+").unwrap();
    static ref GRAMMAR_LITERAL_ESCAPE_RE: Regex = Regex::new(r#"[\r\n"]"#).unwrap();
    static ref GRAMMAR_LITERAL_ESCAPES: HashMap<char, &'static str> = {
        let mut m = HashMap::new();
        m.insert('\r', "\\r");
        m.insert('\n', "\\n");
        m.insert('"', "\\\"");
        m
    };
}

/*
def _build_repetition(
    item_rule, min_items, max_items, separator_rule=None, item_rule_is_literal=False
):
    if not separator_rule:
        if min_items == 0 and max_items == 1:
            return f"{item_rule}?"
        elif min_items == 1 and max_items is None:
            return f"{item_rule}+"

    result = ""

    if min_items > 0:
        if item_rule_is_literal and separator_rule is None:
            result = '"' + (item_rule[1:-1] * min_items) + '"'
        else:
            result = (f" {separator_rule} " if separator_rule else " ").join(
                [item_rule] * min_items
            )

    def opt_repetitions(up_to_n, prefix_with_sep=False):
        """
        - n=4, no sep:             '(a (a (a (a)?)?)?)?'
        - n=4, sep=',', prefix:    '("," a ("," a ("," a ("," a)?)?)?)?'
        - n=4, sep=',', no prefix: '(a ("," a ("," a ("," a)?)?)?)?'
        """

        content = (
            f"{separator_rule} {item_rule}"
            if prefix_with_sep and separator_rule
            else item_rule
        )
        if up_to_n == 0:
            return ""
        elif up_to_n == 1:
            return f"({content})?"
        elif separator_rule and not prefix_with_sep:
            return f"({content} {opt_repetitions(up_to_n - 1, prefix_with_sep=True)})?"
        else:
            return (f"({content} " * up_to_n).rstrip() + (")?" * up_to_n)

    if min_items > 0 and max_items != min_items:
        result += " "

    if max_items is not None:
        result += opt_repetitions(max_items - min_items, prefix_with_sep=min_items > 0)
    else:
        item_operator = f'({separator_rule + " " if separator_rule else ""}{item_rule})'

        if min_items == 0 and separator_rule:
            result = f"({item_rule} {item_operator}*)?"
        else:
            result += f"{item_operator}*"

    return result
*/

fn _build_repetition(
    item_rule: impl AsRef<str>,
    min_items: i32,
    max_items: Option<i32>,
    separator_rule: Option<String>,
    item_rule_is_literal: bool,
) -> String {
    if separator_rule.is_none() {
        if min_items == 0 && max_items == Some(1) {
            return format!("{}?", item_rule.as_ref());
        } else if min_items == 1 && max_items.is_none() {
            return format!("{}+", item_rule.as_ref());
        }
    }

    let mut result = String::new();

    if min_items > 0 {
        if item_rule_is_literal && separator_rule.is_none() {
            result = format!(
                "\"{}\"",
                item_rule.as_ref()[1..item_rule.as_ref().len() - 1].repeat(min_items as usize)
            );
        } else {
            result = if let Some(ref separator_rule) = separator_rule {
                (0..min_items)
                    .map(|_| item_rule.as_ref())
                    .collect::<Vec<_>>()
                    .join(&format!(" {} ", separator_rule))
            } else {
                (0..min_items)
                    .map(|_| item_rule.as_ref())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
        }
    }

    if min_items > 0 && max_items != Some(min_items) {
        result += " ";
    }

    if let Some(max_items) = max_items {
        result += &opt_repititions__build_repetition(
            (max_items - min_items).try_into().unwrap(),
            min_items > 0,
            item_rule.as_ref(),
            separator_rule,
            item_rule_is_literal,
        );
    } else {
        // f'({separator_rule + " " if separator_rule else ""}{item_rule})'
        let item_operator = format!(
            "({}{})",
            if separator_rule.is_some() {
                separator_rule.clone().unwrap() + " "
            } else {
                "".to_string()
            },
            item_rule.as_ref()
        );

        if min_items == 0 && separator_rule.is_some() {
            result = format!("({} {}*)?", item_rule.as_ref(), item_operator);
        } else {
            result += &format!("{}*", item_operator);
        }
    }

    result
}

fn opt_repititions__build_repetition(
    up_to_n: isize,
    prefix_with_sep: bool,
    item_rule: &str,
    separator_rule: Option<String>,
    item_rule_is_literal: bool,
) -> String {
    /*
    - n=4, no sep:             '(a (a (a (a)?)?)?)?'
    - n=4, sep=',', prefix:    '("," a ("," a ("," a ("," a)?)?)?)?'
    - n=4, sep=',', no prefix: '(a ("," a ("," a ("," a)?)?)?)?'
    */
    let content = if prefix_with_sep && separator_rule.is_some() {
        format!("{} {}", &separator_rule.clone().unwrap(), item_rule)
    } else {
        item_rule.to_string()
    };

    if up_to_n == 0 {
        "".to_string()
    } else if up_to_n == 1 {
        format!("({})?", content)
    } else if separator_rule.is_some() && !prefix_with_sep {
        format!(
            "({} {})?",
            content,
            opt_repititions__build_repetition(
                up_to_n - 1,
                true,
                item_rule,
                separator_rule,
                item_rule_is_literal
            )
        )
    } else {
        let rstripped = format!("({} ", content)
            .repeat(up_to_n as usize)
            .trim_end()
            .to_string();
        format!("{}{}", rstripped, ")?".repeat(up_to_n as usize))
    }
}

#[derive(Debug, Clone)]
struct BuiltinRule {
    content: String,
    deps: Vec<String>,
}

impl BuiltinRule {
    fn new(content: impl AsRef<str>) -> Self {
        Self {
            content: content.as_ref().to_string(),
            deps: vec![],
        }
    }
}

fn _up_to_15_digits() -> String {
    _build_repetition("[0-9]", 0, Some(15), None, false)
}

lazy_static! {
    static ref PRIMITIVE_RULES: HashMap<&'static str, BuiltinRule> = {
        let mut m = HashMap::new();
        m.insert("boolean", BuiltinRule::new(r#"("true" | "false") space"#));
        m.insert(
            "decimal-part",
            BuiltinRule::new(format!("[0-9] {}", _up_to_15_digits())),
        );
        m.insert(
            "integral-part",
            BuiltinRule::new(format!("[0-9] | [1-9] {}", _up_to_15_digits())),
        );
        m.insert(
            "number",
            BuiltinRule {
                content:
                    r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space"#
                        .to_string(),
                deps: vec!["integral-part".to_string(), "decimal-part".to_string()],
            },
        );
        m.insert(
            "integer",
            BuiltinRule {
                content: r#"("-"? integral-part) space"#.to_string(),
                deps: vec!["integral-part".to_string()],
            },
        );
        m.insert(
            "value",
            BuiltinRule {
                content: "object | array | string | number | boolean | null".to_string(),
                deps: vec!["object", "array", "string", "number", "boolean", "null"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
        );
        m.insert("object",
        BuiltinRule {
            content: r#""{" space ( string ":" space value ("," space string ":" space value)* )? "}" space"#.to_string(),
            deps: vec!["string".to_string(), "value".to_string()],
        });
        m.insert(
            "array",
            BuiltinRule {
                content: r#""[" space ( value ("," space value)* )? "]" space"#.to_string(),
                deps: vec!["value".to_string()],
            },
        );
        m.insert(
            "uuid",
            BuiltinRule::new(format!(
                r#""\"" {} "\"" space"#,
                [
                    "[0-9a-fA-F]".repeat(8),
                    "[0-9a-fA-F]".repeat(4),
                    "[0-9a-fA-F]".repeat(4),
                    "[0-9a-fA-F]".repeat(4),
                    "[0-9a-fA-F]".repeat(12)
                ]
                .join(" \"-\" ")
            )),
        );
        m.insert("char", BuiltinRule::new(
            r#"[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])"#
        ));
        m.insert(
            "string",
            BuiltinRule {
                content: r#""\"" char* "\"" space"#.to_string(),
                deps: vec!["char".to_string()],
            },
        );
        m.insert("null", BuiltinRule::new(r#""null" space"#));
        m
    };
    static ref STRING_FORMAT_RULES: HashMap<&'static str, BuiltinRule> = {
        let mut m = HashMap::new();
        m.insert("date", BuiltinRule::new(
            r#"[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#
        ));
        m.insert("time", BuiltinRule::new(
            r#"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#
        ));
        m.insert(
            "date-time",
            BuiltinRule {
                content: r#"date "T" time"#.to_string(),
                deps: vec!["date".to_string(), "time".to_string()],
            },
        );
        m.insert(
            "date-string",
            BuiltinRule {
                content: r#""\\"" date "\\"" space"#.to_string(),
                deps: vec!["date".to_string()],
            },
        );
        m.insert(
            "time-string",
            BuiltinRule {
                content: r#""\\"" time "\\"" space"#.to_string(),
                deps: vec!["time".to_string()],
            },
        );
        m.insert(
            "date-time-string",
            BuiltinRule {
                content: r#""\\"" date-time "\\"" space"#.to_string(),
                deps: vec!["date-time".to_string()],
            },
        );
        m
    };
    static ref NON_LITERAL_SET: HashSet<char> = HashSet::from_iter("|.()[]{}*+?".chars());
    static ref ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS: HashSet<char> =
        HashSet::from_iter("[]()|{}*+?".chars());
    static ref RESERVED_NAMES: HashSet<&'static str> = HashSet::from_iter(
        ["root", "dot"]
            .iter()
            .chain(PRIMITIVE_RULES.keys())
            .chain(STRING_FORMAT_RULES.keys())
            .cloned()
    );
}

const DOTALL: &str = "[\\U00000000-\\U0010FFFF]";
const DOT: &str = "[^\\x0A\\x0D]";

fn group_by<T: Clone, F, K>(iter: &[T], key: F) -> Vec<(K, Vec<T>)>
where
    F: Fn(&T) -> K,
    K: PartialEq,
{
    let mut result = Vec::new();

    if let Some((first_key, first_value)) = iter.first().map(|v| (key(v), v.clone())) {
        let mut current_key = first_key;
        let mut current_group = vec![first_value];

        for value in iter.iter().skip(1) {
            let new_key = key(value);

            if new_key == current_key {
                current_group.push(value.clone());
            } else {
                result.push((current_key, current_group));
                current_key = new_key;
                current_group = vec![value.clone()];
            }
        }

        result.push((current_key, current_group));
    }

    result
}

struct SchemaConverter {
    prop_order: HashMap<String, usize>,
    allow_fetch: bool,
    dotall: bool,
    raw_pattern: bool,
    rules: HashMap<String, String>,
    refs: HashMap<String, Value>,
    refs_being_resolved: HashSet<String>,
}

impl SchemaConverter {
    fn new(prop_order: HashMap<String, usize>) -> Self {
        Self {
            prop_order,
            allow_fetch: false,
            dotall: false,
            raw_pattern: false,
            rules: {
                let mut rules = HashMap::new();
                rules.insert("space".to_string(), SPACE_RULE.to_string());
                rules
            },
            refs: HashMap::new(),
            refs_being_resolved: HashSet::new(),
        }
    }

    fn _format_literal(literal: &str) -> String {
        /*
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), literal
        )
        return f'"{escaped}"'
        */
        let escaped = GRAMMAR_LITERAL_ESCAPE_RE.replace_all(literal, |caps: &regex::Captures| {
            GRAMMAR_LITERAL_ESCAPES
                .get(&caps.get(0).unwrap().as_str().chars().next().unwrap())
                .unwrap()
        });
        format!("\"{}\"", escaped)
    }

    // NOT USED IN PYTHON CODE
    // fn not_literal(self, literal: &str, dotall: bool, maybe_escaped_underscores: bool) -> String {
    //     assert!(!literal.is_empty(), "Empty literal not supported");
    //     let mut result = String::from("(");
    //     self.recurse(0, &mut result, literal, maybe_escaped_underscores);
    //     result.push_str(")");

    //     result
    // }

    // fn not_literal_ recurse(self, i: usize, acc: &mut String, literal: &str, maybe_escaped_underscores: bool) {
    //     let c = literal.chars().nth(i).unwrap();
    //     if maybe_escaped_underscores && c == '_' {
    //         acc.push_str(&format!("[^{}\\\\]", c));
    //         acc.push_str(" | ");
    //         acc.push_str(&format!("\"\\\\\"? \"{}\"", c));
    //     } else {
    //         acc.push_str(&format!("[^{}]", c));
    //     }
    //     if i < literal.len() - 1 {
    //         acc.push_str(" | ");
    //         acc.push_str(&Self::_format_literal(&c.to_string()));
    //         acc.push_str(" (");
    //         self.recurse(i + 1, acc, literal, maybe_escaped_underscores);
    //         acc.push_str(")?");
    //     }
    // }

    fn _add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = INVALID_RULE_CHARS_RE.replace_all(name, "_").to_string();

        let key;
        if !self.rules.contains_key(&esc_name) || self.rules.get(&esc_name).unwrap() == rule {
            key = esc_name;
        } else {
            let mut i = 1;
            while self
                .rules
                .contains_key(format!("{}{}", esc_name, i).as_str())
                && self.rules[format!("{}{}", esc_name, i).as_str()] != rule
            {
                i += 1;
            }
            key = format!("{}{}", esc_name, i);
        }
        self.rules.remove(&key);
        self.rules.insert(key.clone(), rule.to_string());
        key
    }

    fn resolve_refs(&mut self, mut schema: Value, url: &str) -> Option<Value> {
        /*
        Resolves all $ref fields in the given schema, fetching any remote schemas,
        replacing $ref with absolute reference URL and populating self._refs with the
        respective referenced (sub)schema dictionaries.
        */
        let schema_clone = schema.clone();
        self.visit_resolve_refs(&mut schema, schema_clone, url)
    }

    fn visit_resolve_refs(&mut self, n: &mut Value, schema: Value, url: &str) -> Option<Value> {
        if n.is_array() {
            return Some(Value::Array(
                n.clone()
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .map(|v| self.visit_resolve_refs(v, schema.clone(), url).unwrap())
                    .collect(),
            ));
        } else if n.is_object() {
            let refv = n.get("$ref");
            if refv.is_some() {
                let mut refv = refv.unwrap().as_str().unwrap().to_string();
                let mut target: Option<Value> = None;
                if refv.starts_with("https://") && !self.refs.contains_key(&refv) {
                    assert!(
                        self.allow_fetch,
                        "Fetching remote schemas is not allowed (use --allow-fetch for force)"
                    );
                    use reqwest;

                    let frag_split: Vec<_> = refv.split("#").collect();
                    let base_url = frag_split[0];

                    let target_response = self.refs.get(base_url);
                    let target = match target_response {
                        Some(target) => target.clone(),
                        None => {
                            let target_schema = serde_json::from_str(
                                &reqwest::blocking::get(base_url).unwrap().text().unwrap(),
                            )
                            .unwrap();
                            let target = self.resolve_refs(target_schema, base_url).unwrap();
                            self.refs.insert(base_url.to_string(), target.clone());
                            target
                        }
                    };

                    if frag_split.len() == 1 || frag_split[frag_split.len() - 1].is_empty() {
                        return Some(target);
                    } else {
                        return None;
                    }
                } else if refv.starts_with("#/") {
                    target = Some(schema.clone());
                    refv = format!("{}{}", url, refv);
                    *n.get_mut("$ref").unwrap() = Value::String(refv.to_string());
                } else {
                    panic!("Unsupported ref: {}", refv);
                }

                for sel in refv.split("#").last().unwrap().split("/").skip(1) {
                    assert!(
                        target.is_some() && target.clone()?.get(sel).is_some(),
                        "Error resolving ref {refv}: {sel} not in {target:?}"
                    );
                    target = Some(target?.get(sel).unwrap().clone());
                }

                self.refs.remove(&refv);
                self.refs.insert(refv, target.clone().unwrap());
                return None;
            } else if let Some(obj) = n.as_object_mut() {
                for v in obj.values_mut() {
                    self.visit_resolve_refs(v, schema.clone(), url);
                }
            }
        }
        Some(n.clone())
    }

    fn _generate_union_rule(&mut self, name: &str, alt_schemas: Vec<Value>) -> String {
        alt_schemas
            .into_iter()
            .enumerate()
            .map(|(i, alt_schema)| {
                self.visit(
                    alt_schema,
                    Some(format!(
                        "{}{}{}",
                        name,
                        if name.is_empty() { "" } else { "-" },
                        i
                    )),
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn _visit_pattern(&mut self, pattern: Regex, name: &str) -> String {
        /*
        Transforms a regular expression pattern into a GBNF rule.

        Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
        Output: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

        Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

        Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
        we define sub-rules to keep the output lean.
        */

        assert!(
            pattern.as_str().starts_with("^") && pattern.as_str().ends_with("$"),
            "Pattern must start with ^ and end with $"
        );
        let pattern = Regex::from_str(
            pattern
                .as_str()
                .trim_start_matches("^")
                .trim_end_matches("$"),
        )
        .unwrap();
        let mut sub_rule_ids: HashMap<String, String> = HashMap::new();

        let i = 0;
        let length = pattern.as_str().len();

        let rule = {
            let pattern = self.transform__visit_pattern(
                pattern,
                name.to_string(),
                length,
                i,
                &mut sub_rule_ids,
            );
            if self.raw_pattern {
                self.to_rule__visit_pattern(pattern)
            } else {
                format!(
                    //  '"\\"" ' + to_rule(transform()) + ' "\\"" space'
                    "\"\\\"\" {rule} \"\\\"\" space",
                    rule = self.to_rule__visit_pattern(pattern)
                )
            }
        };
        self._add_rule(name, &rule)
    }

    fn to_rule__visit_pattern(&mut self, s: (String, bool)) -> String {
        let (txt, is_literal) = s;
        if is_literal {
            format!("\"{}\"", txt)
        } else {
            txt
        }
    }

    fn transform__visit_pattern(
        &mut self,
        pattern: Regex,
        name: String,
        length: usize,
        mut i: usize,
        mut sub_rule_ids: &mut HashMap<String, String>,
    ) -> (String, bool) {
        let start = i;
        let mut seq: Vec<(String, bool)> = vec![];
        let pattern_chars: Vec<char> = pattern.clone().as_str().chars().collect::<Vec<char>>();

        while i < length {
            let c = pattern_chars[i];
            if c == '.' {
                seq.push((self.get_dot__transform__visit_pattern(), false));
                i += 1;
            } else if c == '(' {
                i += 1;
                if i < length {
                    assert!(
                        pattern_chars[i] != '?',
                        "Unsupported pattern syntax \"{}\" at index {} of /{}/",
                        pattern_chars[i],
                        i,
                        pattern.clone().as_str()
                    )
                }

                let pattern_result = self
                    .transform__visit_pattern(
                        pattern.clone(),
                        name.clone(),
                        length,
                        i,
                        &mut sub_rule_ids,
                    )
                    .clone();
                seq.push((
                    format!("({})", self.to_rule__visit_pattern(pattern_result)),
                    false,
                ));
            } else if c == ')' {
                i += 1;
                assert!(
                    start > 0 && pattern_chars[start - 1] == '(',
                    "Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}"
                );
                return self.join_seq__transform__visit_pattern(seq);
            } else if c == '[' {
                let mut square_brackets = c.to_string();
                i += 1;
                while i < length && pattern_chars[i] != ']' {
                    if pattern_chars[i] == '\\' {
                        square_brackets = format!(
                            "{}{}",
                            square_brackets,
                            pattern_chars[i..i + 2].iter().collect::<String>()
                        );
                        i += 2;
                    } else {
                        square_brackets = format!("{}{}", square_brackets, pattern_chars[i]);
                        i += 1;
                    }
                }
                assert!(
                    i < length,
                    "Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}"
                );
                square_brackets = format!("{}{}", square_brackets, pattern_chars[i]);
                i += 1;
                seq.push((square_brackets, false));
            } else if c == '|' {
                seq.push(('|'.to_string(), false));
                i += 1;
            } else if ['*', '+', '?'].contains(&c) {
                let seq_len = seq.len();
                seq[seq_len - 1] = (
                    format!(
                        "{}{}",
                        self.to_rule__visit_pattern(seq[seq_len - 1].clone()),
                        c
                    ),
                    false,
                );
                i += 1;
            } else if c == '{' {
                let mut curly_brackets = c.to_string();
                i += 1;
                while i < length && pattern_chars[i] != '}' {
                    curly_brackets = format!("{}{}", curly_brackets, pattern_chars[i]);
                    i += 1;
                }
                assert!(
                    i < length,
                    "Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}"
                );
                curly_brackets = format!("{}{}", curly_brackets, '}');
                i += 1;
                let nums = curly_brackets[1..curly_brackets.len() - 1]
                    .split(',')
                    .map(|s| s.trim_start().trim_end())
                    .collect::<Vec<&str>>();
                let mut min_times = 0;
                let mut max_times = None;
                if nums.len() == 1 {
                    min_times = nums[0].parse::<i32>().unwrap_or(0);
                    max_times = Some(min_times);
                } else if nums.len() == 2 {
                    min_times = nums[0].parse::<i32>().unwrap_or(0);
                    max_times = nums[1].parse::<i32>().ok();
                } else {
                    panic!(
                        "Invalid quantifier {} in /{}/",
                        curly_brackets,
                        pattern.clone().as_str()
                    );
                }

                let seq_len = seq.len();
                let (mut sub, sub_is_literal) = seq[seq_len - 1].clone();

                if !sub_is_literal {
                    let id = match sub_rule_ids.get(&sub) {
                        Some(id) => id.clone(),
                        None => {
                            let new_id = self
                                ._add_rule(&format!("{}-{}", name, sub_rule_ids.len() + 1), &sub);
                            sub_rule_ids.insert(sub.clone(), new_id.clone());
                            new_id
                        }
                    };
                    sub = id;
                }

                let seq_len = seq.len();
                seq[seq_len - 1] = (
                    _build_repetition(
                        if sub_is_literal {
                            format!("\"{sub}\"")
                        } else {
                            sub
                        },
                        min_times,
                        max_times,
                        None,
                        sub_is_literal,
                    ),
                    false,
                );
            } else {
                let mut literal = "".to_string();
                while i < length {
                    if pattern_chars[i] == '\\' && i < length - 1 {
                        if ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.contains(&pattern_chars[i]) {
                            i += 1;
                            literal = format!("{}{}", literal, pattern_chars[i]);
                            i += 1;
                        } else {
                            literal = format!(
                                "{}{}",
                                literal,
                                pattern_chars[i..i + 2].iter().collect::<String>()
                            );
                            i += 2;
                        }
                    } else if pattern_chars[i] == '"' && !self.raw_pattern {
                        literal = format!("{}{}", literal, "\\\"");
                        i += 1;
                    } else if !NON_LITERAL_SET.contains(&pattern_chars[i])
                        && (i == length - 1
                            || literal.is_empty()
                            || pattern_chars[i + 1] == '.'
                            || !NON_LITERAL_SET.contains(&pattern_chars[i + 1]))
                    {
                        literal = format!("{}{}", literal, pattern_chars[i]);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if !literal.is_empty() {
                    seq.push((literal, true));
                }
            }
        }

        self.join_seq__transform__visit_pattern(seq)
    }

    fn get_dot__transform__visit_pattern(&mut self) -> String {
        let rule = if self.dotall {
            DOTALL.to_string()
        } else {
            DOT.to_string()
        };
        self._add_rule("dot", &rule)
    }

    fn join_seq__transform__visit_pattern(&mut self, seq: Vec<(String, bool)>) -> (String, bool) {
        let mut ret = vec![];
        for (is_literal, g) in group_by(&seq, |x| x.1) {
            if is_literal {
                ret.push((
                    g.iter()
                        .map(|x| x.0.clone())
                        .collect::<Vec<String>>()
                        .join(""),
                    true,
                ));
            } else {
                ret.extend(g);
            }
        }
        if ret.len() == 1 {
            ret[0].clone()
        } else {
            (
                seq.iter()
                    .map(|x| self.to_rule__visit_pattern(x.clone()))
                    .collect::<Vec<String>>()
                    .join(" "),
                false,
            )
        }
    }

    fn _resolve_ref(&mut self, refv: &str) -> String {
        let mut ref_name = refv.split('/').last().unwrap().to_string();
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(refv) {
            self.refs_being_resolved.insert(refv.to_string());
            let resolved = self.refs.get(refv).unwrap();
            ref_name = self.visit(resolved.clone(), Some(ref_name));
            self.refs_being_resolved.remove(refv);
        }
        ref_name
    }

    fn _generate_constant_rule(&mut self, value: &Value) -> String {
        Self::_format_literal(&value.to_string())
    }

    fn visit(&mut self, schema: Value, name: Option<String>) -> String {
        let schema_type = schema.get("type");
        let schema_format = schema.get("format");
        let rule_name = {
            if let Some(ref name) = name {
                if RESERVED_NAMES.contains(name.as_str()) {
                    format!("{}-", name)
                } else {
                    name.to_string()
                }
            } else {
                "root".to_string()
            }
        };

        if schema.get("$ref").is_some() {
            let refv = schema.get("$ref").unwrap();
            let rule = &self._resolve_ref(refv.as_str().unwrap());
            self._add_rule(&rule_name, rule)
        } else if schema.get("oneOf").is_some() || schema.get("anyOf").is_some() {
            let alt_schema = schema.get("oneOf").unwrap_or(schema.get("anyOf").unwrap());
            let rule =
                self._generate_union_rule(name.clone().unwrap().as_str(), vec![alt_schema.clone()]);
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_some_and(|t| t.is_array()) {
            let alt_schemas = schema_type
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|t| json!({"type": t}))
                .collect::<Vec<_>>();
            let rule = self._generate_union_rule(name.clone().unwrap().as_str(), alt_schemas);
            return self._add_rule(&rule_name, &rule);
        } else if schema.get("const").is_some() {
            let rule = self._generate_constant_rule(schema.get("const").unwrap());
            return self._add_rule(&rule_name, &rule);
        } else if schema.get("enum").is_some() {
            let rule = schema
                .get("enum")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| self._generate_constant_rule(v))
                .collect::<Vec<_>>()
                .join(" | ");
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "object")
            && (schema.get("properties").is_some()
                || schema
                    .get("additionalProperties")
                    .is_some_and(|v| !v.as_bool().unwrap()))
        {
            let required = schema
                .get("required")
                .unwrap_or(&json!(Vec::<Vec<String>>::new()))
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<HashSet<String>>();
            let properties = schema
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<Vec<_>>();
            let rule = self._build_object_rule(
                properties,
                required,
                name,
                schema.get("additionalProperties"),
            );
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "object")
            && schema.get("allOf").is_some()
        {
            let mut required = HashSet::new();
            let mut properties = vec![];
            let hybrid_name = name;

            for t in schema.get("allOf").unwrap().as_array().unwrap() {
                if t.get("anyOf").is_some() {
                    for tt in t.get("anyOf").unwrap().as_array().unwrap() {
                        self.add_component__visit(
                            tt.clone(),
                            false,
                            &mut properties,
                            &mut required,
                        );
                    }
                } else {
                    self.add_component__visit(t.clone(), true, &mut properties, &mut required);
                }
            }

            let rule = self._build_object_rule(
                properties,
                required,
                hybrid_name,
                Some(&json!(Vec::<Vec<String>>::new())),
            );
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "array")
            && (schema.get("items").is_some() || schema.get("prefixItems").is_some())
        {
            let items = {
                if schema.get("items").is_some() {
                    schema.get("items").unwrap()
                } else {
                    schema.get("prefixItems").unwrap()
                }
            };
            if items.is_array() {
                let rule = items
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        self.visit(
                            item.clone(),
                            Some(format!(
                                "{}tuple-{}",
                                if name.clone().is_some_and(|n| !n.is_empty()) {
                                    name.clone().unwrap() + "-"
                                } else {
                                    "".to_string()
                                },
                                i
                            )),
                        )
                    })
                    .collect::<Vec<_>>();
                let rule = format!("\"[\" space {} \"]\" space", rule.join(" \",\" space "));
                return self._add_rule(&rule_name, &rule);
            } else {
                let item_rule_name = self.visit(
                    items.clone(),
                    Some(format!(
                        "{}item",
                        if name.clone().is_some_and(|n| !n.is_empty()) {
                            name.clone().unwrap() + "-"
                        } else {
                            "".to_string()
                        }
                    )),
                );
                let min_items = schema
                    .get("minItems")
                    .unwrap_or(&json!(0))
                    .as_i64()
                    .unwrap() as i32;
                let max_items = {
                    if schema.get("maxItems").is_some() {
                        Some(schema.get("maxItems").unwrap().as_i64().unwrap() as i32)
                    } else {
                        None
                    }
                };
                let rep = _build_repetition(
                    item_rule_name,
                    min_items,
                    max_items,
                    Some("\",\" space".to_string()),
                    false,
                );
                let rule = format!("\"[\" space {} \"]\" space", rep);
                return self._add_rule(&rule_name, &rule);
            }
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "string")
            && schema.get("pattern").is_some()
        {
            return self._visit_pattern(
                Regex::new(schema.get("pattern").unwrap().as_str().unwrap()).unwrap(),
                &rule_name,
            );
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "string")
            && regex::Regex::new(r"^uuid[1-5]?$")
                .unwrap()
                .is_match(schema_format.unwrap_or(&json!("")).as_str().unwrap())
        {
            let prim_name = format!("{}-string", schema_format.unwrap().as_str().unwrap());
            let rule = self._add_primitive(
                &prim_name,
                STRING_FORMAT_RULES.get(prim_name.as_str()).unwrap().clone(),
            );
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_some_and(|t| t.as_str().unwrap() == "string")
            && (schema.get("minLength").is_some() || schema.get("maxLength").is_some())
        {
            let char_rule = self._add_primitive("char", PRIMITIVE_RULES["char"].clone());
            let min_len = schema
                .get("minLength")
                .unwrap_or(&json!(0))
                .as_i64()
                .unwrap() as i32;
            let max_len = {
                if schema.get("maxLength").is_some() {
                    Some(schema.get("maxLength").unwrap().as_i64().unwrap() as i32)
                } else {
                    None
                }
            };
            let rule = format!(
                "\"\\\"\" {} \"\\\"\" space",
                _build_repetition(char_rule, min_len, max_len, None, false)
            );
            return self._add_rule(&rule_name, &rule);
        } else if schema_type.is_none_or(|t| t.as_str().unwrap() == "object") {
            let primitive = self._add_primitive("object", PRIMITIVE_RULES["object"].clone());
            return self._add_rule(&rule_name, &primitive);
        } else {
            assert!(PRIMITIVE_RULES.contains_key(schema_type.unwrap().as_str().unwrap()));
            // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_primitive(
                if rule_name == "root" {
                    "root"
                } else {
                    schema_type.unwrap().as_str().unwrap()
                },
                PRIMITIVE_RULES[schema_type.unwrap().as_str().unwrap()].clone(),
            );
        }
    }

    fn add_component__visit(
        &mut self,
        comp_schema: Value,
        is_required: bool,
        properties: &mut Vec<(String, Value)>,
        required: &mut HashSet<String>,
    ) {
        let mut comp_schema_tmp = comp_schema.clone();
        if comp_schema_tmp.get("$ref").is_some() {
            let refv = comp_schema_tmp.get("$ref").unwrap().as_str().unwrap();
            comp_schema_tmp = self.refs.get(refv).unwrap().clone();
        }

        if comp_schema_tmp.get("properties").is_some() {
            for (prop_name, prop_schema) in comp_schema_tmp
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
            {
                properties.push((prop_name.clone(), prop_schema.clone()));
                if is_required {
                    required.insert(prop_name.clone());
                }
            }
        }
    }

    fn _add_primitive(&mut self, name: &str, rule: BuiltinRule) -> String {
        let n = self._add_rule(name, &rule.content);

        for dep in rule.deps {
            let dep = dep.as_str();
            let dep_rule = {
                if PRIMITIVE_RULES.contains_key(dep) {
                    Some(PRIMITIVE_RULES.get(dep).unwrap().clone())
                } else if STRING_FORMAT_RULES.contains_key(dep) {
                    STRING_FORMAT_RULES.get(dep).cloned()
                } else {
                    None
                }
            };
            assert!(dep_rule.is_some(), "Rule {dep} not known");
            if !self.rules.contains_key(dep) {
                self._add_primitive(dep, dep_rule.unwrap());
            }
        }
        n
    }

    fn _build_object_rule(
        &mut self,
        properties: Vec<(String, Value)>,
        required: HashSet<String>,
        name: Option<String>,
        additional_properties: Option<&Value>,
    ) -> String {
        let prop_order = self.prop_order.clone();
        let prop_order_len = prop_order.len();
        // sort by position in prop_order (if specified) then by original order

        let sorted_props = {
            let mut enum_prop = properties.iter().enumerate().collect::<Vec<_>>();
            enum_prop.sort_by(|(ia, kva), (ib, kvb)| {
                if prop_order.get(&kva.0).unwrap_or(&prop_order_len)
                    > prop_order.get(&kvb.0).unwrap_or(&prop_order_len)
                {
                    std::cmp::Ordering::Greater
                } else if ia > ib {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            });
            enum_prop
                .iter()
                .map(|(_, kv)| kv.0.clone())
                .collect::<Vec<_>>()
        };

        let mut prop_kv_rule_names = HashMap::new();
        for (prop_name, prop_schema) in properties {
            let prop_rule_name = self.visit(
                prop_schema,
                Some(format!(
                    "{}{}",
                    if name.clone().is_some_and(|n| !n.is_empty()) {
                        name.clone().unwrap() + "-"
                    } else {
                        "".to_string()
                    },
                    prop_name
                )),
            );
            let rule = &format!(
                "{} space \":\" space {}",
                Self::_format_literal(json!(prop_name).to_string().as_str()),
                prop_rule_name
            );
            prop_kv_rule_names.insert(
                prop_name.clone(),
                self._add_rule(
                    &format!(
                        "{}{}-kv",
                        if name.clone().is_some_and(|n| !n.is_empty()) {
                            name.clone().unwrap() + "-"
                        } else {
                            "".to_string()
                        },
                        prop_name
                    ),
                    rule,
                ),
            );
        }
        let required_props = sorted_props
            .iter()
            .filter(|k| required.contains(*k))
            .cloned()
            .collect::<Vec<_>>();
        let mut optional_props = sorted_props
            .iter()
            .filter(|k| !required.contains(*k))
            .cloned()
            .collect::<Vec<_>>();

        if additional_properties.is_some_and(|v| v.as_bool().is_some_and(|b| b))
            || additional_properties.is_some_and(|v| v.is_object())
        {
            let sub_name = format!(
                "{}additional",
                if name.clone().is_some_and(|n| !n.is_empty()) {
                    name.clone().unwrap() + "-"
                } else {
                    "".to_string()
                }
            );
            let value_rule = self.visit(
                if additional_properties.is_some_and(|v| v.as_bool().is_some_and(|b| b)) {
                    json!({})
                } else {
                    additional_properties.unwrap().clone()
                },
                Some(format!("{}-value", sub_name)),
            );
            let rule = format!(
                "{} \":\" space {}",
                self._add_primitive("string", PRIMITIVE_RULES["string"].clone()),
                value_rule
            );
            prop_kv_rule_names.insert(
                "*".to_string(),
                self._add_rule(&format!("{}-kv", sub_name), &rule),
            );
            optional_props.push("*".to_string());
        }

        let mut rule = "\"{\" space ".to_string();
        rule.push_str(
            &sorted_props
                .iter()
                .map(|k| prop_kv_rule_names.get(k).unwrap().clone())
                .collect::<Vec<_>>()
                .join(" \",\" space "),
        );

        if !optional_props.is_empty() {
            rule = format!("{} (", rule);
            if !required_props.is_empty() {
                rule = format!("{} \",\" space ( ", rule);
            }

            let rule_refs = (0..optional_props.len())
                .map(|i| {
                    Self::get_recursive_refs__build_object_rule(
                        optional_props[i..].to_vec(),
                        false,
                        &mut prop_kv_rule_names,
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            rule = format!("{} {}", rule, rule_refs);
            if !required_props.is_empty() {
                rule = format!("{} )", rule);
            }
            rule = format!("{} )?", rule);
        }

        rule = format!("{} \"}}\" space", rule);

        rule
    }

    fn get_recursive_refs__build_object_rule(
        ks: Vec<String>,
        first_is_optional: bool,
        prop_kv_rule_names: &mut HashMap<String, String>,
    ) -> String {
        "".to_string()
    }

    fn format_grammar(&self) -> String {
        let mut sorted = self
            .rules
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut result = String::new();
        for (name, rule) in sorted.iter() {
            result.push_str(&format!("{} ::= {}\n", name, rule));
        }
        result
    }
}

/*
def json_schema_to_gbnf(schema: str, prop_order: Optional[List[str]] = None):
    prop_order = prop_order or []
    schema = json.loads(schema)
    prop_order = {name: idx for idx, name in enumerate(prop_order)}
    converter = SchemaConverter(
        prop_order=prop_order, allow_fetch=False, dotall=False, raw_pattern=False
    )
    schema = converter.resolve_refs(schema, "stdin")
    converter.visit(schema, "")
    return converter.format_grammar()
*/

pub fn json_schema_to_gbnf(schema: impl AsRef<str>, prop_order: Option<Vec<String>>) -> String {
    let prop_order = prop_order.unwrap_or_default();
    let schema = serde_json::from_str(schema.as_ref()).unwrap();
    let prop_order = prop_order
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect::<HashMap<String, usize>>();
    let mut converter = SchemaConverter::new(prop_order);
    let schema = converter.resolve_refs(schema, "stdin").unwrap();
    converter.visit(schema, None);
    converter.format_grammar()
}

fn main() {
    let s = std::time::Instant::now();
    json_schema_to_gbnf(
        json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "$id": "https://example.com/employee.schema.json",
            "title": "Record of employee",
            "description": "This document records the details of an employee",
            "type": "object",
            "properties": {
                "id": {
                    "description": "A unique identifier for an employee",
                    "type": "number"
                },
                "name": {
                    "description": "name of the employee",
                    "type": "string",
                    "minLength": 2
                },
                "age": {
                    "description": "age of the employee",
                    "type": "number",
                    "minimum": 16
                },
                "hobbies": {
                    "description": "hobbies of the employee",
                    "type": "object",
                    "properties": {
                        "indoor": {
                            "type": "array",
                            "items": {
                                "description": "List of hobbies",
                                "type": "string"
                            },
                            "minItems": 1,
                            "uniqueItems": true
                        },
                        "outdoor": {
                            "type": "array",
                            "items": {
                                "description": "List of hobbies",
                                "type": "string"
                            },
                            "minItems": 1,
                            "uniqueItems": true
                        }
                    },
                    "required": [
                        "indoor",
                        "outdoor"
                    ]
                }
            },
            "required": [
                "id",
                "name",
                "age",
                "hobbies"
            ],
            "additionalProperties": false
        })
        .to_string(),
        None,
    );
    let t = s.elapsed();
    println!("Time: {:?}us", t.as_micros());
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_resolve_refs() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());
        let schema = json! ({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "properties": {
            "name": {
              "type": "string"
            },
            "age": {
              "type": "integer",
              "minimum": 0
            },
            "address": {
              "$ref": "#/definitions/address"
            }
          },
          "required": ["name", "age"],
          "definitions": {
            "address": {
              "type": "object",
              "properties": {
                "street": {
                  "type": "string"
                },
                "city": {
                  "type": "string"
                },
                "postalCode": {
                  "type": "string"
                }
              },
              "required": ["street", "city"]
            }
          }
        }
        );
        let url = "stdin";
        let resolved_schema = schema_converter.resolve_refs(schema.clone(), url);

        assert!(!schema_converter.refs.is_empty());
        assert_eq!(
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "definitions": {
                    "address": {
                        "properties": {
                            "city": {
                                "type": "string"
                            },
                            "postalCode": {
                                "type": "string"
                            },
                            "street": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "street",
                            "city"
                        ],
                        "type": "object"
                    }
                },
                "properties": {
                    "address": {
                        "$ref": "stdin#/definitions/address"
                    },
                    "age": {
                        "minimum": 0,
                        "type": "integer"
                    },
                    "name": {
                        "type": "string"
                    }
                },
                "required": [
                    "name",
                    "age"
                ],
                "type": "object"
            }),
            resolved_schema.unwrap()
        );
    }

    #[test]
    fn test_add_rule() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());
        let rule = DOT;
        let name = "dot";
        let key = schema_converter._add_rule(name, rule);
        println!("key: {:?}", key);
        println!("rules: {:?}", schema_converter.rules);
    }

    #[test]
    fn test_build_repetition() {
        let output = _build_repetition("[0-9]", 0, Some(15), None, false);
        println!("output: {:?}", output);
        assert_eq!(output, "([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9])?)?)?)?)?)?)?)?)?)?)?)?)?)?)?")
    }

    #[test]
    fn test_visit_pattern() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());
        let pattern = Regex::from_str("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$").unwrap();
        let name = "email";
        let key = schema_converter._visit_pattern(pattern, name);
        println!("key: {:?}", key);
        println!("rules: {:#?}", schema_converter.rules);
        println!("rules: {}", schema_converter.rules.get("email").unwrap());
        assert_eq!(key, "email");
        assert_eq!(
            json!(schema_converter.rules),
            json!({
                "email": "\"\\\"\" [a-zA-Z0-9._%+-]+ \"@\" [a-zA-Z0-9.-]+ \"\\.\" email-1 email-1 (email-1)* \"\\\"\" space",
                "email-1": "[a-zA-Z]",
                "space": "\" \"?"
            })
        );
    }

    #[test]
    fn test_build_object_rule() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());

        let properties = vec![
            (
                "tool".to_string(),
                json!({
                    "enum": ["send_message_to_user", "internal_thought", "bash_tool"],
                    "title": "Tool",
                    "type": "string"
                }),
            ),
            (
                "args".to_string(),
                json!({
                    "title": "Args",
                    "type": "string"
                }),
            ),
        ];
        let required = ["tool".to_string(), "args".to_string()]
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<String>>();
        let name = "".to_string();
        let additional_properties = None;
        let rule = schema_converter._build_object_rule(
            properties,
            required,
            Some(name),
            additional_properties,
        );
        assert_eq!(rule, "\"{\" space tool-kv \",\" space args-kv \"}\" space")
    }

    #[test]
    fn test_visit() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());
        let schema = json!({
            "properties": {
                "tool": {
                    "enum": [
                        "send_message_to_user",
                        "internal_thought",
                        "bash_tool"
                    ],
                    "title": "Tool",
                    "type": "string"
                },
                "args": {
                    "title": "Args",
                    "type": "string"
                }
            },
            "required": [
                "args",
                "tool"
            ],
            "title": "Action",
            "type": "object"
        });
        schema_converter.visit(schema, None);
        println!("rules: {:#?}", schema_converter.rules);
        assert_eq!(schema_converter.rules.get("space").unwrap(), "\" \"?");
        assert_eq!(
            schema_converter.rules.get("root").unwrap(),
            "\"{\" space args-kv \",\" space tool-kv \"}\" space"
        );
        assert_eq!(
            schema_converter.rules.get("args-kv").unwrap(),
            "\"\\\"args\\\"\" space \":\" space string"
        );
        assert_eq!(
            schema_converter.rules.get("tool-kv").unwrap(),
            "\"\\\"tool\\\"\" space \":\" space tool"
        );
        assert_eq!(schema_converter.rules.get("char").unwrap(), "[^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])");
        assert_eq!(
            schema_converter.rules.get("string").unwrap(),
            "\"\\\"\" char* \"\\\"\" space"
        );
        assert_eq!(schema_converter.rules.get("tool").unwrap(), "\"\\\"send_message_to_user\\\"\" | \"\\\"internal_thought\\\"\" | \"\\\"bash_tool\\\"\"");
    }

    #[test]
    fn test_visit_complex() {
        let mut schema_converter = SchemaConverter::new(HashMap::new());
        let schema = json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "$id": "https://example.com/employee.schema.json",
            "title": "Record of employee",
            "description": "This document records the details of an employee",
            "type": "object",
            "properties": {
                "id": {
                    "description": "A unique identifier for an employee",
                    "type": "number"
                },
                "name": {
                    "description": "name of the employee",
                    "type": "string",
                    "minLength":2
                },
                "age": {
                    "description": "age of the employee",
                    "type": "number",
                    "minimum": 16
                },
                "hobbies": {
                    "description": "hobbies of the employee",
                    "type": "object",
                    "properties": {
                        "indoor": {
                            "type": "array",
                            "items": {
                                "description": "List of hobbies",
                                "type": "string"
                            },
                            "minItems": 1,
                            "uniqueItems": true
                        },
                        "outdoor": {
                            "type": "array",
                            "items": {
                                "description": "List of hobbies",
                                "type": "string"
                            },
                            "minItems": 1,
                            "uniqueItems": true
                        }
                    },
                    "required": [
                        "indoor",
                        "outdoor"
                    ]
                }
            },
            "required": [
                "id",
                "name",
                "age",
                "hobbies"
            ],
         "additionalProperties": false
        });
        schema_converter.visit(schema, None);
        println!("rules: {:#?}", schema_converter.rules);
        let correct_rules = json!({
            "age-kv": "\"\\\"age\\\"\" space \":\" space number",
            "char": "[^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])",
            "decimal-part": "[0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9])?)?)?)?)?)?)?)?)?)?)?)?)?)?)?",
            "hobbies": "\"{\" space hobbies-indoor-kv \",\" space hobbies-outdoor-kv \"}\" space",
            "hobbies-indoor": "\"[\" space string (\",\" space string)* \"]\" space",
            "hobbies-indoor-kv": "\"\\\"indoor\\\"\" space \":\" space hobbies-indoor",
            "hobbies-kv": "\"\\\"hobbies\\\"\" space \":\" space hobbies",
            "hobbies-outdoor": "\"[\" space string (\",\" space string)* \"]\" space",
            "hobbies-outdoor-kv": "\"\\\"outdoor\\\"\" space \":\" space hobbies-outdoor",
            "id-kv": "\"\\\"id\\\"\" space \":\" space number",
            "integral-part": "[0-9] | [1-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9] ([0-9])?)?)?)?)?)?)?)?)?)?)?)?)?)?)?",
            "name": "\"\\\"\" char char (char)* \"\\\"\" space",
            "name-kv": "\"\\\"name\\\"\" space \":\" space name",
            "number": "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space",
            "root": "\"{\" space age-kv \",\" space hobbies-kv \",\" space id-kv \",\" space name-kv \"}\" space",
            "space": "\" \"?",
            "string": "\"\\\"\" char* \"\\\"\" space"
        });
        for (k, v) in correct_rules.as_object().unwrap() {
            println!("key: {}", k);
            assert_eq!(schema_converter.rules.get(k).unwrap(), v, "key: {}", k);
        }
    }

    #[test]
    fn test_json_schema_to_gbnf() {
        let schema = json!({
            "properties": {
                "tool": {
                    "enum": [
                        "send_message_to_user",
                        "internal_thought",
                        "bash_tool"
                    ],
                    "title": "Tool",
                    "type": "string"
                },
                "args": {
                    "title": "Args",
                    "type": "string"
                }
            },
            "required": [
                "tool",
                "args"
            ],
            "title": "Action",
            "type": "object"
        });
        let output = json_schema_to_gbnf(schema.to_string(), None);
        println!("{}", output);
    }
}
