import { createContext, useContext } from "react";
import type { BookPreferences } from "../../services/ReadingPreferencesService";
import type { useLearningPage } from "./controller";
export interface LearningPresentation { book: BookPreferences; controller: ReturnType<typeof useLearningPage> }
export const LearningContext = createContext<LearningPresentation | null>(null);
export const useLearningPresentation = () => useContext(LearningContext);
export const OrdinaryTranslationContext = createContext(true);
export const useOrdinaryTranslations = () => useContext(OrdinaryTranslationContext);
