from abc import ABC, abstractclassmethod
import pathlib
import imghdr
import os
from typing import Union, Tuple, List, Callable, Any, Dict

class dataLoaderABC(ABC):
    @abstractclassmethod
    def __init__(self, file_directory: Union[str, pathlib.Path, os.DirEntry], fileType: str='image'):
        self.fileType = fileType if fileType in ["image", 'csv', 'excel'] else None
        self.file_directory: Union[str, pathlib.Path, os.DirEntry] = file_directory
        self.file_list: List[pathlib.Path] = []
        self.file_filter: Callable = lambda: True

    @abstractclassmethod
    def LoadFileList(self, file_type:str=None) -> List[pathlib.Path]:
        pass

    @abstractclassmethod
    def check_file_is_image(
        self,
        file_path: Union[str, pathlib.Path, os.DirEntry],
        footer: Union[str, List[str], Tuple[str]]=None
    ) -> bool:
        pass


class dataLoader(dataLoaderABC):
    def __init__(self,file_directory: Union[str, pathlib.Path, os.DirEntry]):
        super().__init__(file_directory)
    
    def LoadFileList(self, file_type:str=None, fileNameList:List=None) -> List[pathlib.Path]:
        dir_path = pathlib.Path(self.file_directory)
        if not dir_path.is_dir():
            raise TypeError(f'Not a valid directory: {dir_path}')
        
        if self.fileType is not None:  
            if self.fileType == "image": self.file_filter = self.check_file_is_image
        
        if fileNameList is None:
            self.file_list = [
                pathlib.Path(file) for file in os.scandir(dir_path) if self.file_filter(file)
            ]
        else:
            for fileName in fileNameList:
                file_path = pathlib.Path(dir_path) / fileName
                if file_path.exists() and self.file_filter(file_path):
                    self.file_list.append(file_path)
        return self.file_list

    def check_file_is_image(
        self,
        file_path: Union[str, pathlib.Path, os.DirEntry],
        footer: Union[str, List[str], Tuple[str]]='jpg'
    ) -> bool:
        if not pathlib.Path(file_path).is_file():
            return False
        file_type = imghdr.what(file_path)
        if file_type is None:
            return False
        if footer is not None:
            return pathlib.Path(file_path).name.lower().endswith(footer)
        return True
