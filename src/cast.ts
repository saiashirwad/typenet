/**
 * The shape assertion, in a module of its own.
 *
 * `assertChecked` is defined next to the `*Check` types in `shape.ts` —
 * that is where its contract has to be readable — but it is re-exported
 * here so that every place in the library that *asserts* a shape rather
 * than deriving one is one grep away:
 *
 *     rg 'from "\.\./src/cast\.ts"'
 *
 * Use it only where the shape is true but not derivable (a `narrow` cut
 * to a runtime width, a permute the algebra cannot follow), never to get
 * past a `*Check` that fired.
 */
export { assertChecked } from "./shape.ts"
