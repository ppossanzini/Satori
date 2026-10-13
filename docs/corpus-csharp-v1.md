# C# Coding Corpus v1

Status: DW02 proposal, pending DW03 approval. Frozen task set for the coding-quality comparison (protocol
section 3). No candidate was scored against it yet.

## Rules

- Each task is a prompt plus an automated criterion: the produced code compiles under the pinned SDK
  (.NET 10) and passes the provided test.
- Pass rate = tasks passing / tasks total. Numerator and denominator are reported for every candidate.
- The prompts below are never used for quantization calibration.
- Different models or quantizations are compared by task success, not token identity.

## C# tasks (English prompts)

| Id | Prompt (abridged) | Criterion |
|---|---|---|
| c01 | `public static int Add(int a, int b)` | sum correct for positive, negative and zero |
| c02 | `public static bool IsPrime(int n)` | true for 2, 3, 17; false for 1, 0, negatives, 4, 9 |
| c03 | `public static string Reverse(string s)` | correct for empty, ASCII and non-BMP Unicode |
| c04 | `public static long Factorial(int n)` | 0→1, 5→120, 20→2432902008176640000 |
| c05 | `public static IReadOnlyList<int> Fibonacci(int n)` | n=0 empty, n=1 [0], n=7 [0,1,1,2,3,5,8] |
| c06 | `public static bool IsPalindrome(string s)` ignoring case and punctuation | sentence palindrome true; non-palindrome false |
| c07 | `public static T Max<T>(IEnumerable<T> xs) where T : IComparable<T>` | correct maximum; throws on empty |
| c08 | `public sealed class Stack<T>` with Push, Pop, Peek, Count | LIFO order; Pop on empty throws `InvalidOperationException` |
| c09 | `public static int BinarySearch(int[] a, int target)` | found index; −1 when absent; empty array |
| c10 | `public static string ToRoman(int n)` for 1..3999 | 4→IV, 9→IX, 1994→MCMXCIV |
| c11 | `public static Dictionary<string,int> WordCount(string text)` case-insensitive, punctuation ignored | expected counts on a fixed string |
| c12 | `public static async Task<int> SumIntegersAsync(string path)` skips blanks | fixed file sums correctly; missing file throws I/O error |

## Italian subset (Italian prompts)

| Id | Prompt (abridged) | Criterion |
|---|---|---|
| i01 | Media aritmetica di una lista, eccezione se vuota | correct value; throws on empty list |
| i02 | Separatore delle migliaia italiano (1234567 → 1.234.567) | correct grouping and sign |
| i03 | Validità di una data nel formato gg/MM/aaaa | valid and invalid dates classified correctly |
| i04 | Ordinamento di stringhe italiane senza distinguere maiuscole o accenti | expected order on a fixed list |

## Long output

At least the long-form budget (section 5 of the protocol) of generated tokens, to expose repetition,
truncation and context loss.

| Id | Prompt (abridged) | Criterion |
|---|---|---|
| l01 | Complete C# console Caesar cipher with encrypt/decrypt, argument parsing and error handling | compiles; encrypt/decrypt round-trips |
| l02 | Small C# JSON parser (objects, arrays, strings, numbers, true/false/null) | compiles; parses a fixed document |

## Fixtures

Layout per task: `fixtures/corpus/csharp-v1/<id>/prompt.txt` (the exact prompt), `Test.cs` (the xUnit test),
`meta.json` (id, language, difficulty, criterion). A runner compiles the model output with the test and
records compile/run success. Materialised when DW03 approves the set.

## Sampling

Deterministic reference runtimes use temperature 0 (greedy) with the pinned context. A stochastic
configuration, if used, fixes the seed. Both are stated in every report.
