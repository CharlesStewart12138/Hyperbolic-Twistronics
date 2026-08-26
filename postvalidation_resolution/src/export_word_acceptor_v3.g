LoadPackage("kbmag");;
outfile := GAPInfo.SystemEnvironment.POSTVALIDATION_WORD_ACCEPTOR;;
F := FreeGroup("a","b","c","d");;
a := F.1;; b := F.2;; c := F.3;; d := F.4;;
G := F / [a*b^-1*c*d^-1*a^-1*b*c^-1*d];;
R := KBMAGRewritingSystem(G);;
ok := AutomaticStructure(R);;
if not ok then Error("KBMAG automatic structure failed"); fi;
wa := WordAcceptor(R);;
table := DenseDTableFSA(wa);;
initial := wa.initial[1];;
inverses := [];;
for symbol in [1..wa.alphabet.size] do
  oneLetterState := table[initial][symbol];;
  forbidden := Filtered([1..wa.alphabet.size], j -> table[oneLetterState][j] = 0);;
  if Length(forbidden) <> 1 then
    Error("could not derive a unique inverse from the one-letter state");
  fi;
  Add(inverses,forbidden[1]);;
od;
out := OutputTextFile(outfile,false);;
SetPrintFormattingStatus(out,false);;
AppendTo(out,"GAP_VERSION=",GAPInfo.Version,"\n");
AppendTo(out,"AUTOMATIC=true\n");
AppendTo(out,"STATE_COUNT=",wa.states.size,"\n");
AppendTo(out,"ALPHABET_SIZE=",wa.alphabet.size,"\n");
AppendTo(out,"INITIAL=",JoinStringsWithSeparator(List(wa.initial,String),","),"\n");
AppendTo(out,"ACCEPTING=",JoinStringsWithSeparator(List(wa.accepting,String),","),"\n");
AppendTo(out,"ALPHABET=",JoinStringsWithSeparator(List(R!.alphabet,String),"|"),"\n");
AppendTo(out,"INVERSE_INDEX=",JoinStringsWithSeparator(List(inverses,String),","),"\n");
for state in [1..wa.states.size] do
  AppendTo(out,"ROW=",state,"|",JoinStringsWithSeparator(List(table[state],String),","),"\n");
od;
CloseStream(out);;
Print("AUTOMATIC=true\n");
Print("STATE_COUNT=",wa.states.size,"\n");
Print("INVERSE_INDEX=",inverses,"\n");
QUIT;
