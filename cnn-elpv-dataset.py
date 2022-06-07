#!/usr/bin/env python3
"""CNN-based identification of defective solar cells in electroluminescence imagery.

@author: Arne Ludwig <arne.ludwig@posteo.de>
@copyright: Copyright © 2022 by Arne Ludwig
"""

from elpv_dataset.utils.elpv_reader import load_dataset

images, probs, types = load_dataset()
