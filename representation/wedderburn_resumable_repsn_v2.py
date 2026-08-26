from __future__ import annotations

from pathlib import Path

from representation import wedderburn_resumable as implementation
from representation.wedderburn_resumable import B07StageFailure


def repsn_irrep_script(index: int, target: Path) -> str:
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SizeScreen([1000000,1000000]);;",
            f"idx:={index};;",
            'Print("STAGE_BEGIN=individual_irrep_repsn\\n");',
            'if LoadPackage("repsn")<>true then Error("Repsn package unavailable"); fi;',
            'Print("IRREP_BACKEND=Repsn.IrreducibleAffordingRepresentation\\n");',
            'Print("IRREP_INDEX=",idx,"\\n");',
            'Print("IRREP_DEGREE=",B07_IRR[idx][1],"\\n");',
            "irrepStart:=Runtime();;",
            "rep:=IrreducibleAffordingRepresentation(B07_IRR[idx]);;",
            'Print("IRREP_CONSTRUCTION_MS=",Runtime()-irrepStart,"\\n");',
            "verifyStart:=Runtime();;",
            "affordingVerified:=IsAffordingRepresentation(B07_IRR[idx],rep);;",
            'Print("AFFORDING_VERIFICATION_MS=",Runtime()-verifyStart,"\\n");',
            'Print("AFFORDING_VERIFIED=",affordingVerified,"\\n");',
            'if affordingVerified<>true then Error("representation does not afford requested irreducible character"); fi;',
            "d:=DimensionOfMatrixGroup(Image(rep));;",
            'if d<>B07_IRR[idx][1] then Error("representation degree mismatch"); fi;',
            f'out:=OutputTextFile("{implementation.to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"REP_BEGIN index=",idx," degree=",d,"\\n");',
            "for g in [1..Length(B07_GENS)] do",
            "  mf:=Image(rep,B07_GENS[g]);; mi:=Image(rep,B07_GENS[g]^-1);;",
            "  expected:=B07_IRR[idx][PositionProperty(ConjugacyClasses(B07_G),c->B07_GENS[g] in c)];;",
            '  PrintTo(out,"TRACE_CHECK generator=",g," equal=",TraceMat(mf)=expected,"\\n");',
            "  for r in [1..d] do for c in [1..d] do",
            "    if not IsZero(mf[r][c]) then",
            '      PrintTo(out,"GEN_ENTRY rep=",idx," generator=",g," inverse=false row=",r," col=",c," value=",String(mf[r][c]),"\\n");',
            "    fi;",
            "    if not IsZero(mi[r][c]) then",
            '      PrintTo(out,"GEN_ENTRY rep=",idx," generator=",g," inverse=true row=",r," col=",c," value=",String(mi[r][c]),"\\n");',
            "    fi;",
            "  od; od;",
            "od;",
            'PrintTo(out,"REP_END\\n");',
            "CloseStream(out);;",
            'Print("STAGE_COMPLETE=individual_irrep_repsn\\n");',
            "QUIT;",
            "",
        ]
    )


def install() -> None:
    implementation._irrep_script = repsn_irrep_script


def prepare_wedderburn(*args, **kwargs):
    install()
    return implementation.prepare_wedderburn(*args, **kwargs)


def run(*args, **kwargs):
    return implementation.run(*args, **kwargs)
