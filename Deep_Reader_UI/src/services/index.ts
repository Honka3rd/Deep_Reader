import { DocumentCatalogService } from "./DocumentCatalogService";
import { DocumentPreparationService } from "./DocumentPreparationService";
import { ManualStructureService } from "./ManualStructureService";
import { ReadingInteractionService } from "./ReadingInteractionService";
import { RestClient } from "./RestClient";
import { StructureRepairService } from "./StructureRepairService";
import { TaskLayoutService } from "./TaskLayoutService";
import { TaskUnitContentService } from "./TaskUnitContentService";

const restClient = new RestClient();

export const documentCatalogService = new DocumentCatalogService(restClient);
export const documentPreparationService = new DocumentPreparationService(restClient);
export const manualStructureService = new ManualStructureService(restClient);
export const readingInteractionService = new ReadingInteractionService(restClient);
export const taskLayoutService = new TaskLayoutService(restClient);
export const taskUnitContentService = new TaskUnitContentService(restClient);
export const structureRepairService = new StructureRepairService(restClient);

export {
  DocumentCatalogService,
  DocumentPreparationService,
  ManualStructureService,
  ReadingInteractionService,
  RestClient,
  StructureRepairService,
  TaskLayoutService,
  TaskUnitContentService,
};
