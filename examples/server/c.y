%{
#include <stdio.h>
#include <string.h>

void yyerror(const char *s);
int yylex(void);
%}

%token STRING NUMBER TRUE FALSE NUL WHITESPACE

%%

root
    : object
    ;

value
    : object
    | array
    | STRING
    | NUMBER
    | TRUE
    | FALSE
    | NUL
    ;

object
    : '{' '}'
    | '{' members '}'
    ;

members
    : pair
    | members ',' pair
    ;

pair
    : STRING ':' value
    ;

array
    : '[' ']'
    | '[' elements ']'
    ;

elements
    : value
    | elements ',' value
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}

int main(void) {
    return yyparse();
}