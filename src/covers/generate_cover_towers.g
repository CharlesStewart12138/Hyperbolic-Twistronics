# Exact GAP declaration for three inequivalent abelian finite-cover towers.
# Usage: gap -q src/covers/generate_cover_towers.g
F := FreeGroup("a1", "b1", "a2", "b2");;
a1 := F.1;; b1 := F.2;; a2 := F.3;; b2 := F.4;;
Gamma := F / ([a1,b1] * [a2,b2]);;

for p in [2,3,5] do
  m := p;
  Q := AbelianGroup([m,m,m,m]);;
  hom := GroupHomomorphismByImages(
    Gamma, Q, GeneratorsOfGroup(Gamma), GeneratorsOfGroup(Q)
  );
  K := Kernel(hom);
  cosets := RightCosets(Gamma, K);
  action := Action(Gamma, cosets, OnRight);
  Print("tower=abelian_", p,
        " level=1 index=", Index(Gamma,K),
        " genus=", 1+Index(Gamma,K),
        " normal=", IsNormal(Gamma,K),
        " action_degree=", NrMovedPoints(action), "\n");
od;
QUIT;

