# 
# This file is part of the share-cv distribution (https://github.com/CheneyYin/share-cv).
# Copyright (c) 2024 Chengyu Yan.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import os

_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path(model_name: str):
    return os.path.join(_root_path, 'models', model_name)

def get_sample_path(sample_name: str, kind='videos'):
    return os.path.join(_root_path, 'samples', kind, sample_name)