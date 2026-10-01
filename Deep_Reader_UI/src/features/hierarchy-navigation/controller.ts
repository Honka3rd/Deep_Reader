import { useMemo } from "react";
import type { DocumentTaskLayout } from "../../types/api";
import { countSections, countTaskUnits } from "./model";

export function useHierarchyNavigationController(layout: DocumentTaskLayout | null) {
  return useMemo(
    () => ({
      sectionCount: countSections(layout),
      unitCount: countTaskUnits(layout),
    }),
    [layout],
  );
}
