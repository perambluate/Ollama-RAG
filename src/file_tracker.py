import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DocumentInfo:
    file: str
    timestamp: float
    ids: List[str]

class FileTracker:
    def __init__(self, track_file: str):
        self.track_file = track_file
        if not os.path.exists(track_file):
            if not os.path.exists(os.path.dirname(track_file)):
                os.makedirs(os.path.dirname(track_file))
            self.tracking_data = []
            self.tracking_dict = {}
        else:
            with open(self.track_file, 'r') as f:
                self.tracking_data = json.load(f)
            self.tracking_dict = {f['file']:
                                {'timestamp': f['timestamp'], 'ids': f['ids']} for f in self.tracking_data}

    def get_modified_files(self, directory: str) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        """
        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]: Three dictionaries containing
                - new_files: {file_path: timestamp} for new files
                - modified_files: {file_path: timestamp} for modified files
                - deleted_files: [file_path] for deleted files
        """
        new_files = {}
        modified_files = {}
        existing_files = set()

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                existing_files.add(file_path)
                mod_time = os.path.getmtime(file_path)

                if file_path not in self.tracking_dict:
                    new_files[file_path] = mod_time
                elif mod_time != self.tracking_dict[file_path]['timestamp']:
                    modified_files[file_path] = mod_time
        
        # Find deleted files
        deleted_files = [f for f in self.tracking_dict.keys() if f not in existing_files]

        return new_files, modified_files, deleted_files

    def update_tracking(self, documents_to_update: List[DocumentInfo] = None, files_to_remove: List[str] = None) -> None:
        """Update tracking data for files"""
        if not documents_to_update and not files_to_remove:
            return
        
        # Remove files
        if files_to_remove:
            for file_path in files_to_remove:
                self.tracking_dict.pop(file_path, None)
        
        # Update files with their timestamps
        if documents_to_update:
            for doc in documents_to_update:
                self.tracking_dict[doc['file']] = dict(
                    timestamp=doc['timestamp'],
                    ids=doc['ids']
                )

        self.tracking_data = [
            dict(file=f, timestamp=info['timestamp'], ids=info['ids']) for f, info in self.tracking_dict.items()
        ]

        # Save to file
        with open(self.track_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=4)