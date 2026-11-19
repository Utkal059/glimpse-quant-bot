/**
 * Branded unit types for the API boundary.
 *
 * The engine never mixes money and quotes, and neither does the UI:
 *   - Millisats:   integer millisatoshis (1 sat = 1000 msat). Money, always.
 *   - PriceUnits:  the 0-100 LS-LMSR quote scale (a contract settles at 100).
 *   - Probability: a probability in [0, 1].
 *
 * The brands exist only at compile time, so a `Millisats` cannot be passed where
 * a `PriceUnits` is expected (or vice versa) without an explicit conversion.
 * Arithmetic on money stays integer: values are only divided for display.
 */

declare const brand: unique symbol;
type Brand<T, B extends string> = T & { readonly [brand]: B };

export type Millisats = Brand<number, "Millisats">;
export type PriceUnits = Brand<number, "PriceUnits">;
export type Probability = Brand<number, "Probability">;

export const MSAT_PER_SAT = 1000;

export function millisats(value: number): Millisats {
  if (!Number.isSafeInteger(value)) {
    throw new RangeError(`Millisats must be a safe integer, got ${value}`);
  }
  return value as Millisats;
}

export function priceUnits(value: number): PriceUnits {
  if (!Number.isFinite(value)) throw new RangeError(`PriceUnits must be finite, got ${value}`);
  return value as PriceUnits;
}

export function probability(value: number): Probability {
  if (!Number.isFinite(value)) throw new RangeError(`Probability must be finite, got ${value}`);
  return value as Probability;
}

/** Integer-safe sum of money values. */
export function sumMillisats(values: Iterable<Millisats>): Millisats {
  let total = 0;
  for (const value of values) total += value;
  return millisats(total);
}

/** Display-only: sats as a float. Never feed the result back into money maths. */
export function toSats(value: Millisats): number {
  return value / MSAT_PER_SAT;
}

/** A probability expressed on the quote scale (p * 100 points). */
export function probabilityToPoints(p: Probability): PriceUnits {
  return priceUnits(p * 100);
}

/** A quote read as an implied probability (price / 100). */
export function pointsToProbability(p: PriceUnits): Probability {
  return probability(p / 100);
}
