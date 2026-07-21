# Vendored Boost headers

This directory contains the Boost header subset used to compile the bundled
TecIO sources. It was generated from Boost 1.88.0 with the official
`bcp --scan` tool and is distributed
under the Boost Software License 1.0 in `LICENSE_1_0.txt`.

Only headers are vendored: no Boost binary library is built or linked. The
dependency closure includes macro-generated and platform-specific headers so
GCC and Clang builds work without a system Boost installation.
