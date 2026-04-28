#!/bin/bash

mkdir -p docs

# Remove pre-existing docs
find docs -type f -name "*.html" -delete || true


package="sftk"
# Generate the base package documentation and copy to index.html so it can be launched as a http server
python -m pydoc -w "$package"
# Sed command removes links to the source code
sed -i 's|<a href="[^"]*\.py">[^<]*</a>||g' "$package.html"
cp -f "$package.html" "docs/index.html"
# Move the original so linking works
mv -f "$package.html" "docs/"



# TODO: This will only work for the first level of modules, need to make it recursive if subdirectories are added
# Generate the documentation for each module and move to the docs folder
for module in $package/*.py; do
    file_no_ext="$(basename "$module" .py)"
    python -m pydoc -w "$package.$file_no_ext"
    html_doc="$package.$file_no_ext.html"
    sed -i 's|<a href="[^"]*\.py">[^<]*</a>||g' "$html_doc"
    mv -f "$html_doc" "docs/"
done
