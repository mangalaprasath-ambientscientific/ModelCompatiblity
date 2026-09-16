import { useState, useEffect ,  forwardRef, useImperativeHandle  } from "react";
import { Folder, ChevronDown, ChevronRight } from "lucide-react";
import { Plus, FolderPlus, RefreshCw, Minimize2 } from "lucide-react";
import "react-confirm-alert/src/react-confirm-alert.css";
import "../styles/FolderStructure.css";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { fetchPreprocessingSteps } from "../components/api";
import { customPrompt, customConfirm } from "../components/CustomConfirm";

const FolderItem = ({ item, contextMenu, setContextMenu, refreshStructure, onItemChanged, onFileSelect, fullRefreshCallback, folderType  }) => {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleToggle = async () => {
    if (item.type === "file" && item.name.endsWith(".py")) {
      onFileSelect && onFileSelect(item.path);
      return;
    }

    if (!expanded && item.type === "folder" && children.length === 0) {
      const res = await fetch(`http://127.0.0.1:9262/api/folder-structure?path=${encodeURIComponent(item.path)}&type=${folderType}`);
      const data = await res.json();
      setChildren(data);
    }
    setExpanded(!expanded);
  };

  const handleContextMenu = (e) => {
    e.preventDefault();
    e.stopPropagation(); // Prevent event bubbling up to parent folders

    setContextMenu({
      x: e.pageX,
      y: e.pageY,
      targetPath: item.path,
      targetType: item.type
    });

    // Clear context menu on next click anywhere
    document.addEventListener("click", () => setContextMenu(null), { once: true });
  };

  const refreshFolder = async () => {
    if (item.type === "folder") {
      setIsRefreshing(true);
      try {
        const res = await fetch(`http://127.0.0.1:9262/api/folder-structure?path=${encodeURIComponent(item.path)}&type=${folderType}`);
        const data = await res.json();
        setChildren(data);
      } catch (error) {
        console.error("Error refreshing folder:", error);
      } finally {
        setIsRefreshing(false);
      }
    }
  };

  // Just the updated rename handler function
  const handleRename = async () => {
    customPrompt({
      title: "Rename Item",
      message: "Enter new name:",
      defaultValue: item.name,
      onConfirm: async (newName) => {
        if (!newName || newName === item.name) return;

        try {
          // Get the full path without any basename manipulation
          const currentPath = item.path;

          // Send the full current path and new name to the server
          const response = await fetch("http://127.0.0.1:9262/api/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              path: currentPath,
              new_name: newName
            }),
          });



          // First refresh the parent folder
          if (onItemChanged) {
            onItemChanged();
          }

          // Then refresh the entire structure to ensure everything is updated
          refreshStructure();
        } catch (error) {
          console.error("Rename error:", error);
          alert(`Rename failed: ${error.message}`);
        }
      },
    });
  };

  const handleDelete = () => {
    customConfirm({
      title: "Confirm to delete",
      message: "Are you sure you want to delete the file:",
      itemName: item.name,
      onConfirm: async () => {
        try {
          const response = await fetch("http://127.0.0.1:9262/api/delete", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: item.path }),
          });

          const result = await response.json();

          if (response.ok) {
            if (onItemChanged) onItemChanged(); // Call only after success
            refreshStructure();
          } else {
            console.error("Delete failed:", result.error);
          }
        } catch (error) {
          console.error("Delete request error:", error);
        }
      },
    });
  };


  const handleNewItem = async (type) => {
    // Use the exact path of the clicked folder as the parent path
    const parentPath = contextMenu.targetPath;


    customPrompt({
      title: `Create ${type === "file" ? "File" : "Folder"}`,
      message: `Enter ${type} name:`,
      onConfirm: async (name) => {
        if (!name) return;

        if (type === "file") {
          const parts = name.split(".");
          const extension = parts.length > 1 ? parts.pop() : null;

          if (!extension) {
            toast.error("Filename must include an extension (e.g., main.py).");
            return;
          }

          const allowedExtensions = ["py"];
          if (!allowedExtensions.includes(extension)) {
            toast.error("Only .py and .txt files are allowed.");
            return;
          }

        }

        try {
          // Send the full parentPath to the server
          const response = await fetch(`http://127.0.0.1:9262/api/create-${type}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name,
              parent_path: parentPath
            }),
          });



          // Force expand the folder and refresh its contents
          if (!expanded) {
            setExpanded(true);
          }

          // Refresh the children of this folder
          const res = await fetch(`http://127.0.0.1:9262/api/folder-structure?path=${encodeURIComponent(parentPath)}`);
          const data = await res.json();
          setChildren(data);

          // Also refresh the entire structure to ensure all views are updated
          if (fullRefreshCallback) {
            fullRefreshCallback();
          } else {
            refreshStructure(); 
          }
        } catch (error) {
          console.error(`Error creating ${type}:`, error);
        }
      },
    });
  };

  return (
    <div className="folder-item">
      <div
        className="item-header"
        onClick={handleToggle}
        onContextMenu={handleContextMenu}
      >
        {item.type === "folder" ? (
          <>
            {expanded ? <ChevronDown className="icon chevron" /> : <ChevronRight className="icon chevron" />}
            <Folder className="icon folder-icon" />
          </>
        ) : (
          getFileIcon(item.name)
        )}

        <span className="item-name">{item.name}</span>
      </div>

      {expanded && children.length > 0 && (
        <div className="children">
          {children.map((child, idx) => (
            <FolderItem
              key={`${child.path}-${idx}`}
              item={child}
              contextMenu={contextMenu}
              setContextMenu={setContextMenu}
              refreshStructure={refreshStructure}
              onItemChanged={refreshFolder}
              onFileSelect={onFileSelect}
              fullRefreshCallback={fullRefreshCallback}
            />
          ))}
        </div>
      )}

      {contextMenu?.targetPath === item.path && (
        <div
          className="context-menu"
          style={{ top: contextMenu.y, left: contextMenu.x }}
        >
          <div onClick={handleRename}>Rename</div>
          <div onClick={handleDelete}>Delete</div>
          {/* Only show new file/folder options for folders */}
          {item.type === "folder" && (
            <>
              <div onClick={() => handleNewItem("file")}>New File</div>
              <div onClick={() => handleNewItem("folder")}>New Folder</div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

const getFileIcon = (filename) => {
  const ext = filename.split(".").pop();
  const iconMap = {
    js: "FileCode",
    ts: "FileCode",
    html: "FileHtml",
    css: "FileText",
    json: "FileJson",
    py: "FileCode",
    md: "FileText",
    txt: "FileText",
    default: "File"
  };
  const Icon = require("lucide-react")[iconMap[ext] || iconMap.default];
  return <Icon className="icon file-icon" />;
};

const FolderStructure = forwardRef(({ onFileSelect, setLeftItems, setRightItems, folderType = 0  }, ref) => {
  const [root, setRoot] = useState([]);
  const [contextMenu, setContextMenu] = useState(null);

  useEffect(() => {
    fetchRootStructure();
  }, [folderType]); 

  const fetchRootStructure = () => {
    fetch(`http://127.0.0.1:9262/api/folder-structure?type=${folderType}`)
      .then((res) => res.json())
      .then((data) => setRoot(data));
  };

  useImperativeHandle(ref, () => ({
    handleRefresh: () => {
        fetchRootStructure();
        try {
            const fetchPreprocessingStepsAsync = async () => {
                const steps = await fetchPreprocessingSteps();
                if (setLeftItems) {
                    setLeftItems(steps);
                }

                if (setRightItems) {
                  setRightItems([]); 
                }
            };
            fetchPreprocessingStepsAsync();
        } catch (error) {
            console.error("Failed to fetch preprocessing steps:", error);
            toast.error("Failed to refresh preprocessing steps");
        }
    }
}));

  const handleRefresh = async () => {
    fetchRootStructure();
    try {
      const steps = await fetchPreprocessingSteps();
      setLeftItems(steps);
    } catch (error) {
      console.error("Failed to fetch preprocessing steps:", error);
    }
  };

  const handleCollapseAll = () => {
    setRoot([]);
    setTimeout(handleRefresh, 100);
  };

  const handleFullRefresh = async () => {
    fetchRootStructure();
    try {
      const steps = await fetchPreprocessingSteps(false); 
      if (setLeftItems) {
        setLeftItems(steps);
      }
      if (setRightItems) {
        setRightItems([]); // Reset right items
      }
    } catch (error) {
      console.error("Failed to fetch preprocessing steps:", error);
      toast.error("Failed to refresh preprocessing steps");
    }
  };

  const createItem = (type) => {
    customPrompt({
      title: `Create ${type === "file" ? "File" : "Folder"}`,
      message: `Enter ${type} name:`,
      onConfirm: (name) => {
        if (!name) return;

        if (type === "file") {
          const parts = name.split(".");
          const extension = parts.length > 1 ? parts.pop() : null;

          if (!extension) {
            toast.error("Filename must include an extension (e.g., main.py).");
            return;
          }

          const allowedExtensions = ["py"];
          if (!allowedExtensions.includes(extension)) {
            toast.error("Only .py files are allowed.");
            return;
          }

        }

        fetch(`http://127.0.0.1:9262/api/create-${type}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            parent_path: ""
          }),
        })
          .then(() => handleFullRefresh());

      },
    });
  };

  return (
    <div className="folder-structure-container">
      <div className="folder-header">
        <span className="folder-title">Directory</span>
        <div className="folder-toolbar">
          <button title="New File" className="toolbar-btn" onClick={() => createItem("file")}><Plus size={16} /></button>
          <button title="New Folder" className="toolbar-btn" onClick={() => createItem("folder")}><FolderPlus size={16} /></button>
          <button title="Refresh" className="toolbar-btn" onClick={handleRefresh}><RefreshCw size={16} /></button>
          <button title="Collapse All" className="toolbar-btn" onClick={handleCollapseAll}><Minimize2 size={16} /></button>
        </div>
      </div>

      <div className="folder-structure">
        {root.map((item, idx) => (
          <FolderItem
            key={`root-${item.path}-${idx}`}
            item={item}
            contextMenu={contextMenu}
            setContextMenu={setContextMenu}
            refreshStructure={handleRefresh}
            onFileSelect={onFileSelect}
            fullRefreshCallback={handleFullRefresh}
          />
        ))}
      </div>
    </div>
  );
});

export default FolderStructure;