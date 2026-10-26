import { forwardRef, type ReactNode } from "react";
import {
  TreeItemContent,
  TreeItemGroupTransition,
  TreeItemIconContainer,
  TreeItemLabel,
  TreeItemRoot,
} from "@mui/x-tree-view/TreeItem";
import { TreeItemIcon } from "@mui/x-tree-view/TreeItemIcon";
import { TreeItemProvider } from "@mui/x-tree-view/TreeItemProvider";
import { useTreeItem } from "@mui/x-tree-view/useTreeItem";

interface NavigationTreeItemProps {
  itemId: string;
  label: string;
  copy: ReactNode;
  action: ReactNode;
  insight?: ReactNode;
  children?: ReactNode;
  className: string;
  contentClassName?: string;
  disabled?: boolean;
}

// Keep menus and inline insights separate from the selectable label while MUI
// owns tree registration, roving focus, keyboard navigation and expansion.
export const NavigationTreeItem = forwardRef<HTMLLIElement, NavigationTreeItemProps>(
  function NavigationTreeItem({ itemId, label, copy, action, insight, children,
    className, contentClassName = "", disabled }, ref) {
    const { getRootProps, getContentProps, getLabelProps, getIconContainerProps,
      getGroupTransitionProps, getContextProviderProps, status } = useTreeItem({
      itemId, label, children, disabled, rootRef: ref,
    });

    return (
      <TreeItemProvider {...getContextProviderProps()}>
        <TreeItemRoot {...getRootProps({ className, "data-item-id": itemId,
          "aria-label": label, "aria-selected": status.selected })}>
          <TreeItemContent {...getContentProps({
            className: `hierarchy-tree-content ${contentClassName}`,
          })} status={status}>
            <TreeItemIconContainer {...getIconContainerProps({ className: "hierarchy-tree-icon" })}>
              <TreeItemIcon status={status} />
            </TreeItemIconContainer>
            <TreeItemLabel {...getLabelProps({ className: "hierarchy-tree-label" })}>
              {copy}
            </TreeItemLabel>
            <div className="hierarchy-tree-action" onClick={event => event.stopPropagation()}
              onMouseDown={event => event.stopPropagation()} onKeyDown={event => event.stopPropagation()}>
              {action}
            </div>
          </TreeItemContent>
          {insight && <div className="hierarchy-tree-insight" onKeyDown={event => event.stopPropagation()}>
            {insight}
          </div>}
          {children && <TreeItemGroupTransition {...getGroupTransitionProps({
            className: "hierarchy-tree-group", timeout: 0,
          })} />}
        </TreeItemRoot>
      </TreeItemProvider>
    );
  },
);
