#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/setup.sh"

show_help() {
cat << EOF
Usage: ${0##*/} [-h] -m {train,test} -x XP_NUM -v FN_V -e EMB
Perform frame identification using Electra-AGE_FE.

  -h, --help                            display this help and exit
  -m, --mode                            train or test on all models 
  -x, --xp       XP_NUM                 xp number written as 3 digits (e.g. 001)
  -v, --fnv      FN_V                   FramNet version either 1.5 or 1.7 
  -e, --embs     EMB                    Embeddings to be used (bert/electra)
EOF
}

is_xp_set=FALSE
is_mode_set=FALSE
is_ver_set=FALSE
is_embs_set=FALSE

while :; do
    case $1 in
        -h|-\?|--help)
            show_help
            exit
            ;;
        -x|--xp)
            if [ "$2" ]; then
                is_xp_set=TRUE
                xp="xp_$2"
                shift
            else
                die "ERROR: '--xp' requires a non-empty option argument"
            fi
            ;;
        -v|--fnv)
            if [ "$2" ]; then
                is_ver_set=TRUE
                fnv=$2
                shift
            else
                die "ERROR: '--fnv' requires a non-empty option argument"
            fi
            ;;
        -e|--embs)
            if [ "$2" ]; then
                is_embs_set=TRUE
                embs=$2
                shift
            else
                die "ERROR: '--embs' requires a non-empty option argument"
            fi
            ;;
        -m|--mode)
            if [ "$2" ]; then
                is_mode_set=TRUE
                mode=$2
                shift
            else
                die "ERROR: '--mode' requires a non-empty option argument"
            fi
            ;;
        --)
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)
            break
    esac
    shift
done

if [ "${is_ver_set}" = FALSE ]; then
    fnv="1.5"
fi
echo "Frame Identification for FN v"${fnv}

if [ "${is_embs_set}" = FALSE ]; then
    embs="bert"
fi
echo "Using as embeddings "${embs}

if [ "${is_xp_set}" = FALSE ]; then
    die "ERROR: '--xp' parameter is required."
fi

if [ "${is_mode_set}" = FALSE ]; then
    die "ERROR: '--mode' parameter is required."
fi


prepare2() {
  rm -f ${XP_DIR}/${xp}/frameid/data/lexicons/fn${fnv}_All*
  python3 ${SRC_DIR}/generateLex.py ${XP_DIR}/${xp}/frameid/data/corpora/train.frame.elements ${XP_DIR}/${xp}/frameid/data/lexicons/fnTrain_lexicon
  cp ${XP_DIR}/../resources/fn${fnv}_lexicon ${XP_DIR}/${xp}/frameid/data/lexicons/
  cat ${XP_DIR}/${xp}/frameid/data/lexicons/fn*_lexicon | sort | uniq > ${XP_DIR}/${xp}/frameid/data/lexicons/joined_lex
  cp ${XP_DIR}/../resources/fn${fnv}_All* ${XP_DIR}/${xp}/frameid/data/lexicons/
  cp ${XP_DIR}/../resources/fn${fnv}.Frame-FE.pkl ${XP_DIR}/${xp}/frameid/data/lexicons/
  cp -R ${XP_DIR}/../resources/AGE_data ${XP_DIR}/${xp}/frameid/data/lexicons/
}

if [ "${mode}" = train ]; then
  bash ./prepare.sh ${xp}
  prepare2
  echo "Training frame identification on all models using ${parser}..."
  python3 ${SRC_DIR}/main.py train ${XP_DIR}/${xp}/frameid ${embs} fn${fnv}
  echo "Done"
else
  bash ./prepare.sh ${xp}
  prepare2
  echo "Test frame identification on all models using ${parser}..."
  python3 ${SRC_DIR}/main.py test ${XP_DIR}/${xp}/frameid ${embs} fn${fnv}
  echo "Done"
fi
