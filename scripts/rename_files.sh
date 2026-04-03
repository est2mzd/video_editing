

cd /workspace/src/parse/

BEFORE_KEYWORD="prototype_instruction"
AFTER_KEYWORD="instruction"
EXTENSION=".py"

DO_RENAME=true


# Check Rename
for f in *"$BEFORE_KEYWORD"*"$EXTENSION"; do
  echo mv "$f" "${f/$BEFORE_KEYWORD/$AFTER_KEYWORD}"
done

# Perform Rename
if [ "$DO_RENAME" = true ]; then
  for f in *"$BEFORE_KEYWORD"*"$EXTENSION"; do
    mv "$f" "${f/$BEFORE_KEYWORD/$AFTER_KEYWORD}"
  done
fi