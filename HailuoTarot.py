# Made by JimWang for ComfyUI
# 02/04/2023
import os
import torch
import random

from torchvision.transforms.functional import to_tensor, to_pil_image
from typing import List
from PIL import Image
import time
import random
from PIL import Image
from typing import List, Tuple
import shutil
import subprocess
import platform

class TarotCard:
    def __init__(self, name: str, description: str, meaning: str, image_path: str):
        self.name = name
        self.description = description
        self.meaning = meaning
        self.image_path = image_path  # 图像路径

class TarotDeck:
    def __init__(self):
        self.cards = self.create_deck()

    def create_deck(self):
        #大阿尔克那牌，共有22张牌（从0到21），象征重大事件或生命旅程中的重要阶段。
        # 愚者、魔术师、女祭司、女皇、皇帝、教皇、恋人、战车、力量、隐者、命运之轮、正义、倒吊人、死神、节制、恶魔、塔、星星、月亮、太阳、审判、世界
        major_arcana = [
            # TarotCard("愚者", "新的开始、冒险", "象征无畏的探索和自由原始的冲动", "img/major_arcana/愚者.jpg"),
            # TarotCard("魔法师", "能力、资源", "展现你的创造力，使用手头的工具", "img/major_arcana/魔法师.jpg"),
            # TarotCard("女祭司", "直觉、潜意识", "倾听内心，揭示潜藏的真理", "img/major_arcana/女祭司.jpg"),
            # TarotCard("女皇", "培养、滋养", "自然的母性，关怀的形象", "img/major_arcana/女皇.jpg"),
            # TarotCard("皇帝", "结构、权力", "象征稳定和控制的父性形象", "img/major_arcana/皇帝.jpg"),
            # TarotCard("教皇", "传统、信仰", "代表道德和精神指导", "img/major_arcana/教皇.jpg"),
            # TarotCard("恋人", "爱情、选择", "象征关系中的和谐与选择", "img/major_arcana/恋人.jpg"),
            # TarotCard("战车", "决心、胜利", "代表控制和意志力的胜利", "img/major_arcana/战车.jpg"),
            # TarotCard("力量", "内在的力量、勇气", "象征内心的力量和勇气", "img/major_arcana/力量.jpg"),
            # TarotCard("隐者", "内省、孤独", "寻求真理和内心的智慧", "img/major_arcana/隐者.jpg"),
            # TarotCard("命运之轮", "循环、变革", "象征命运的起伏和变化", "img/major_arcana/命运之轮.jpg"),
            # TarotCard("正义", "公平、真理", "代表公正和因果法则", "img/major_arcana/正义.jpg"),
            # TarotCard("吊人", "牺牲、放下", "代表不同的视角与放下", "img/major_arcana/吊人.jpg"),
            # TarotCard("死神", "转变、结束", "象征结束与新生的转变", "img/major_arcana/死神.jpg"),
            # TarotCard("节制", "平衡、和谐", "代表节制与内心的和谐", "img/major_arcana/节制.jpg"),
            # TarotCard("恶魔", "诱惑、物质", "象征物质欲望和束缚", "img/major_arcana/恶魔.jpg"),
            # TarotCard("塔", "突发事件、崩溃", "象征意外的改变与启示", "img/major_arcana/塔.jpg"),
            # TarotCard("星星", "希望、灵感", "代表希望与精神的宁静", "img/major_arcana/星星.jpg"),
            # TarotCard("月亮", "直觉、神秘", "揭示潜意识的梦境，警惕幻象的出现", "img/major_arcana/月亮.jpg"),
            # TarotCard("太阳", "活力、成功", "快乐、幸福和生命的象征", "img/major_arcana/太阳.jpg"),
            # TarotCard("审判", "复活、觉醒", "象征反思与新生的机会", "img/major_arcana/审判.jpg"),
            # TarotCard("世界", "完成、成就", "一切都到达了终点，成就感及满足感", "img/major_arcana/世界.jpg")
            TarotCard("The Fool", "New Beginnings, Adventure",
                      "Symbolizes fearless exploration and the raw impulse of freedom",
                      "img/major_arcana/The_Fool.jpg"),
            TarotCard("The Magician", "Ability, Resources", "Showcase your creativity and use the tools at hand",
                      "img/major_arcana/The_Magician.jpg"),
            TarotCard("The High_Priestess", "Intuition, Subconscious",
                      "Listen to your inner voice and reveal hidden truths", "img/major_arcana/The_High_Priestess.jpg"),
            TarotCard("The Empress", "Nurturing, Growth", "The embodiment of natural maternal care and abundance",
                      "img/major_arcana/The_Empress.jpg"),
            TarotCard("The Emperor", "Structure, Authority", "Symbolizes stability and the controlling father figure",
                      "img/major_arcana/The_Emperor.jpg"),
            TarotCard("The Hierophant", "Tradition, Faith", "Represents moral and spiritual guidance",
                      "img/major_arcana/The_Hierophant.jpg"),
            TarotCard("The Lovers", "Love, Choice", "Symbolizes harmony in relationships and making choices",
                      "img/major_arcana/The_Lovers.jpg"),
            TarotCard("The Chariot", "Determination, Victory", "Represents triumph through control and willpower",
                      "img/major_arcana/The_Chariot.jpg"),
            TarotCard("Strength", "Inner Strength, Courage", "Symbolizes inner fortitude and bravery",
                      "img/major_arcana/Strength.jpg"),
            TarotCard("The Hermit", "Introspection, Solitude", "Seeking truth and inner wisdom",
                      "img/major_arcana/The_Hermit.jpg"),
            TarotCard("Wheel of Fortune", "Cycles, Change", "Symbolizes the ups and downs of fate",
                      "img/major_arcana/Wheel_of_Fortune.jpg"),
            TarotCard("Justice", "Fairness, Truth", "Represents justice and the law of cause and effect",
                      "img/major_arcana/Justice.jpg"),
            TarotCard("The Hanged_Man", "Sacrifice, Letting Go", "Represents a new perspective and surrender",
                      "img/major_arcana/The_Hanged_Man.jpg"),
            TarotCard("Death", "Transformation, Endings", "Symbolizes the transition from endings to new beginnings",
                      "img/major_arcana/Death.jpg"),
            TarotCard("Temperance", "Balance, Harmony", "Represents moderation and inner peace",
                      "img/major_arcana/Temperance.jpg"),
            TarotCard("The Devil", "Temptation, Materialism", "Symbolizes material desires and bondage",
                      "img/major_arcana/The_Devil.jpg"),
            TarotCard("The Tower", "Sudden Upheaval, Chaos", "Symbolizes unexpected change and revelation",
                      "img/major_arcana/The_Tower.jpg"),
            TarotCard("The Star", "Hope, Inspiration", "Represents hope and spiritual tranquility",
                      "img/major_arcana/The_Star.jpg"),
            TarotCard("The Moon", "Intuition, Mystery", "Reveals subconscious dreams and warns of illusions",
                      "img/major_arcana/The_Moon.jpg"),
            TarotCard("The Sun", "Vitality, Success", "Symbolizes joy, happiness, and life",
                      "img/major_arcana/The_Sun.jpg"),
            TarotCard("Judgement", "Rebirth, Awakening", "Symbolizes reflection and new opportunities",
                      "img/major_arcana/Judgement.jpg"),
            TarotCard("The World", "Completion, Achievement",
                      "Everything has reached fulfillment, bringing a sense of accomplishment and satisfaction",
                      "img/major_arcana/The_World.jpg")
        ]

        # 小阿尔克那，由56张牌组成，分为四个花色：杯（Cups）、剑（Swords）、权杖（Wands）和星币（Pentacles），反映日常生活的事件。
        # suits = ["杯", "剑", "权杖", "星币"]
        suits = ["Cup", "Sword", "Wand", "Pentacle"]
        minor_arcana = []
        for suit in suits:
            for i in range(1, 11):
                image_name = f"{suit}_{i}.jpg"  # 假设小阿尔克那牌按此命名
                # minor_arcana.append(TarotCard(f"{suit} {i}", f"{suit} 的 {i}", f"象征着 {suit} 的力量和意图", f"img/minor_arcana/{image_name}"))
                minor_arcana.append(
                    TarotCard(f"{suit} {i}", f"{suit} of {i}", f"Symbolizes the power and intention of {suit}",
                              f"img/minor_arcana/{image_name}"))

            minor_arcana.append(TarotCard(f"{suit} Page", f"Page of {suit}", f"The beginning and exploration of {suit}",
                                          f"img/minor_arcana/{suit}_Page.jpg"))
            minor_arcana.append(TarotCard(f"{suit} Knight", f"Knight of {suit}", f"The action and pursuit of {suit}",
                                          f"img/minor_arcana/{suit}_Knight.jpg"))
            minor_arcana.append(TarotCard(f"{suit} Queen", f"Queen of {suit}", f"The wisdom and care of {suit}",
                                          f"img/minor_arcana/{suit}_Queen.jpg"))
            minor_arcana.append(TarotCard(f"{suit} King", f"King of {suit}", f"The control and leadership of {suit}",
                                          f"img/minor_arcana/{suit}_King.jpg"))

        return major_arcana + minor_arcana

    def shuffle(self):
        random.seed(time.time())
        random.shuffle(self.cards)

    # 1、单牌抽取：适合简短的回答或解决一个明确的问题。
    # 2、三张牌：通常表示过去、现在和未来，适合多角度思考。
    # 3、十字塔（CelticCross）：较为复杂，能够深入分析问题及其背景。
    def draw(self, num_cards: int):
        drawn_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]  # 更新剩余牌堆
        return drawn_cards


# 读取随机收取的塔罗牌解析文本信息
def load_text(drawn_cards: List[TarotCard]) -> str:
    return "\n".join([f"{card.name}" for card in drawn_cards])
    # return "\n".join([f"{card.name}: {card.description}\nAnalysis: {card.meaning}" for card in drawn_cards])

# 读取随机抽取的塔罗牌图片结果集


def load_image(drawn_cards: List[TarotCard],path='') -> Tuple[List[Image.Image], List[int]]:
    """
    加载塔罗牌图像，并根据时间戳随机决定是否翻转图像。
    :param drawn_cards: 塔罗牌对象列表。
    :return: 返回一个元组，包含：
             - 图像列表（List[Image.Image]）。
             - 翻转标志列表（List[int]），1 表示翻转，0 表示未翻转。
    """
    # 初始化图像列表和翻转标志列表
    image_list = []
    flip_flags = []
    # 设置随机种子为当前时间戳，确保每次运行结果不同
    random.seed(os.times().elapsed)  # 使用系统时间作为随机种子

    for card in drawn_cards:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, card.image_path)

        # 检查文件是否存在
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGBA")  # 确保图像为 RGBA 格式
        else:
            print(f"文件 {card.image_path} 不存在于 data 文件夹中。")
            file_path = os.path.join(current_dir, 'img/default.png')
            img = Image.open(file_path).convert("RGBA")  # 加载默认图像

        # 根据 35% 的概率决定是否翻转图像
        random.seed(time.time())
        if random.random() < 0.35:  # random.random() 生成 [0, 1) 之间的随机数
            print(f'是否历史牌path：{path}')
            if path=='':
                img = img.rotate(180)  # 将图像旋转 180 度
                flip_flags.append(1)  # 记录翻转标志为 1
            else:
                flip_flags.append(0)  # 记录翻转标志为 0
            print(f'是否反转flip_flags[0]：{flip_flags[0]}')
        else:
            flip_flags.append(0)  # 记录翻转标志为 0

        # 将图像添加到列表中
        image_list.append(img)

    return image_list, flip_flags

 # 显示图片的函数
def image_stitch(images):

    if not images:
        raise ValueError("输入的图像列表不能为空。")

    # 将所有图像转换为 Tensor
    image_tensors = [to_tensor(img) for img in images]

    # 检查所有图像的高度是否一致
    heights = [img.shape[1] for img in image_tensors]
    if len(set(heights)) != 1:
        raise ValueError("所有图像的高度必须一致。")

    # 横向拼接图像
    stitched_tensor = torch.cat(image_tensors, dim=2)  # 在宽度维度上拼接

    # 将 Tensor 转换回 PIL.Image
    stitched_image = to_pil_image(stitched_tensor)
    return stitched_image


class TarotDealCard:

    def __init__(self):
        # 解析 XML 文件
        self.supported_font_extensions = [".ttf"]
        self.destination_font_dir = "/usr/share/fonts/"  # 硬编码字体安装目标目录

        # 获取当前目录下的 font 文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fonts_dir = os.path.join(current_dir, "font")

        self.install_font_batch()  # 自动安装字体

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "card_num": ("INT", {"default":1}),
                "text": ("STRING", {"default":''}),
                "cur_round": ("STRING", {"default":'1'}),
                "all_flag": ("STRING", {"default": '0'}),
                "image_path_1": ("STRING", {"default":''}),
                "image_path_2": ("STRING", {"default":''}),
                "image_path_3": ("STRING", {"default":''}),
                "card1": ("STRING", {"default": '0'}),
                "card2": ("STRING", {"default": '0'}),
                "card3": ("STRING", {"default": '0'}),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("Card_NO_1","Card_NO_2","Card_NO_3","CARD_DESC_1","CARD_DESC_2","CARD_DESC_3","CAR_NAME_1","CAR_NAME_2","CAR_NAME_3","cur_round","all_card_name")
    FUNCTION = "card_deal"
    OUTPUT_NODE = True
    CATEGORY = "HailuoTarot"
    DESCRIPTION = "tarot card deal"

    def card_deal(self, card_num: int,image_path_1,image_path_2,image_path_3, card1,card2,card3,text='', cur_round="1", all_flag="0"):
        """
        根据输入的参数进行塔罗牌抽取、图像处理和文本生成。
        :param card_num: 每轮抽取的塔罗牌数量。
        :param text: 附加文本（未使用）。
        :param cur_round: 当前轮次，默认为 "1"。
        :param all_flag: 是否处理所有轮次，默认为 "0"。
        :return: 返回一个字典，包含处理后的图像和文本。
        """
        # 将输入参数 cur_round 和 all_flag 转换为数字
        all_card_name=""
        try:
            cur_round = int(cur_round)
            all_flag = int(all_flag)
        except ValueError:
            raise ValueError("cur_round 和 all_flag 必须是数字")

        # 校验 cur_round 和 all_flag 的合理性
        if cur_round < 1 or cur_round > 3:
            raise ValueError("cur_round 必须在 1 到 3 之间")
        if all_flag not in (0, 1):
            raise ValueError("all_flag 必须是 0 或 1")

        # 当 all_flag = 1 时，cur_round = 3
        if all_flag == 1:
            cur_round = 3
        print(f"i当前参数:img_path_1:{image_path_1},img_path_2:{image_path_2},image_path_3:{image_path_3}")
        print(f"i当前参数:card1:{card1},card2:{card2},card3:{card3}")
        # 初始化返回的图像和文本
        result = {
            "images": [self.get_blank_img() for _ in range(3)],
            "texts": ["" for _ in range(3)],
            "names": ["" for _ in range(3)]
        }
        result["texts"][0] = card1
        result["texts"][1] = card2
        result["texts"][2] = card3

        if card1 is None or card1 == "":
            result["images"][0]=self.get_blank_img()
        else:
            img, txt, name = self.load_card_and_text(image_path_1)
            tar = self.convert_to_target_format(img)
            result["images"][0] = tar
            result["texts"][0] = txt
            result["names"][0] = name
            
        if card2 is None or card2 == "":
            result["images"][1]=self.get_blank_img()
        else:
            img, txt, name = self.load_card_and_text(image_path_2)
            tar = self.convert_to_target_format(img)
            result["images"][1] = tar
            result["texts"][1] = txt
            result["names"][1] = name
            
        if card3 is None or card3 == "":
            result["images"][2]=self.get_blank_img()
        else:
            img, txt, name = self.load_card_and_text(image_path_3)
            tar = self.convert_to_target_format(img)
            result["images"][2] = tar
            result["texts"][2] = txt
            result["names"][2] = name
            
        if cur_round==1:
            img, txt,name = self.load_card_and_text(image_path_1)
            print(f"curround:1{image_path_1},name{name}")
            tar = self.convert_to_target_format(img)
            result["images"][0]=tar
            result["texts"][0]=txt
            result["names"][0]=name
            all_card_name=all_card_name+txt+","
        if cur_round==2:
            img, txt,name = self.load_card_and_text(image_path_1)
            print(f"curround:2{image_path_1},name:{name}")
            tar = self.convert_to_target_format(img)
            result["images"][0]=tar
            result["texts"][0]=txt
            result["names"][0]=name
            all_card_name = all_card_name + txt + ","

            img, txt,name = self.load_card_and_text("")
            print(f"curround:3：image_path_2：{image_path_2},name{name}")
            tar = self.convert_to_target_format(img)
            result["images"][1] = tar
            result["texts"][1]= txt
            result["names"][1]=name
            all_card_name = all_card_name + txt + ","

        if cur_round == 3:
            img, txt,name = self.load_card_and_text(image_path_1)

            tar = self.convert_to_target_format(img)
            result["images"][0]=tar
            result["texts"][0]=txt
            result["names"][0]=name
            all_card_name = all_card_name + txt + ","

            img, txt,name = self.load_card_and_text(image_path_2)

            tar = self.convert_to_target_format(img)
            result["images"][1] = tar
            result["texts"][1]= txt
            result["names"][1]=name
            all_card_name = all_card_name + txt + ","
            img, txt,name = self.load_card_and_text(image_path_3)

            tar = self.convert_to_target_format(img)
            result["images"][2] = tar
            result["texts"][2]= txt
            result["names"][2]=name
            all_card_name = all_card_name + txt + ","
        # 循环执行塔罗牌相关作业
        print(f"all flag{all_flag}")
        if all_flag==1:
            for i in range(cur_round):
            # 抽取塔罗牌对象
            # 转换为目标格式
                img,txt,name=self.load_card_and_text("")
                tar = self.convert_to_target_format(img)
                result["images"][i] = tar
                result["texts"][i] = txt
                result["names"][i] = name
                all_card_name = all_card_name + txt + ","

        # 返回结果

        return(result["images"][0], result["images"][1], result["images"][2],\
            result["texts"][0], result["texts"][1], result["texts"][2],\
            result["names"][0], result["names"][1], result["names"][2],cur_round,all_card_name)

    def load_card_and_text(self,path=''):
        isReversed = False
        print(f"Current path value:{path}")
        if path!='':
            # 根据路径读取图片，判断是否是翻转的图片
            if "Reversed" in path:
                isReversed = True
                path = path.replace("Reversed", "")  # 替换翻转图片路径

        drawn_cards = self.load_tarotCard(1,path)
        name=drawn_cards[0].image_path
        images, flags = load_image(drawn_cards,path)
        print(f'flags[0]:{flags[0]}')
        print(f'isReversed：{isReversed}，{isReversed == True}')
        # 翻转图片
        if isReversed == True:            
            images[0] = images[0].rotate(180)  # 将图像旋转 180 度
            flags[0] = "1" #重置标识

        txt= load_text(drawn_cards)
        # 合并塔罗牌图片
        txt = txt.rstrip(',') if txt.endswith(',') else txt
        img = image_stitch(images)

        if str(flags[0]) == "1":
            txt=txt + "(Reversed)"
            name="Reversed"+name  #标记是翻转图片的路径

        return img,txt,name,

    # 随机读取塔罗牌对象列表
    def load_tarotCard(self, card_nums: int, path='') -> List[TarotCard]:
        tarotDeck = TarotDeck()
        tarotDeck.shuffle()
        if card_nums <= 0:
            card_nums = 1
        drawn_cards = tarotDeck.draw(card_nums)
        # 根据路径抽排
        if path != '':
            for tarCar in tarotDeck.create_deck():
                if tarCar.image_path == path:
                    drawn_cards[0] = tarCar
                    return drawn_cards

        return drawn_cards

    def convert_to_target_format(self,image):
        """
        将图像转换为目标格式 [1, H, W, C]
        """
        if isinstance(image, torch.Tensor):
            return image
        tensor = to_tensor(image)  # 转换为 [C, H, W]
        tensor = tensor.permute(1, 2, 0)  # 转换为 [H, W, C]
        tensor = tensor.unsqueeze(0)  # 添加 batch 维度，变为 [1, H, W, C]
        return tensor

    def get_blank_img(self) :

        # 获取当前目录下的 img 文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'img')

        file_name="blank.png"


        file_path = os.path.join(data_dir, file_name)
        result_img=Image.open(file_path).convert("RGBA")
        result_img = self.convert_to_target_format(result_img)
        return result_img

    def validate_font_file(self, font_path):
        """
        验证字体文件路径和文件类型
        """
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件 {font_path} 不存在！")

        file_extension = os.path.splitext(font_path)[1]
        if file_extension.lower() not in self.supported_font_extensions:
            raise ValueError(
                f"支持的字体类型为：{'、'.join(self.supported_font_extensions)}，当前文件为 '{file_extension}'！")

    def check_font_installed(self, font_path):
        """
        检查字体文件是否已安装
        """
        self.validate_font_file(font_path)  # 先验证字体文件

        font_file_name = os.path.basename(font_path)
        font_dirs = [self.destination_font_dir, os.path.expanduser("~/.fonts/")]  # 常见的字体目录
        for directory in font_dirs:
            if directory and os.path.exists(directory):
                if font_file_name in os.listdir(directory):
                    return True
        return False

    def install_font(self, font_path):
        """
        将字体文件安装到目标目录
        """
        self.validate_font_file(font_path)  # 验证字体文件

        destination_path = os.path.join(self.destination_font_dir, os.path.basename(font_path))

        # 检查目标目录是否存在，不存在则创建
        if not os.path.exists(self.destination_font_dir):
            os.makedirs(self.destination_font_dir)

        # 复制字体文件到目标目录
        shutil.copy2(font_path, destination_path)

        return destination_path

    @staticmethod
    def refresh_font_cache():
        """
        刷新字体缓存（跨平台）
        """
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["fc-cache", "-f"], capture_output=True, check=True)
                print("字体缓存刷新完成！")
            elif system == "Windows":
                subprocess.run(["powershell", "Start-Process", "C:\\Windows\\System32\\control.exe", "-ArgumentList",
                                "'C:\\Windows\\Fonts'"], check=True)
                print("字体缓存刷新完成！")
            else:
                print(f"不支持的操作系统：{system}，无法刷新字体缓存！")
        except subprocess.CalledProcessError as e:
            print(f"刷新字体缓存失败：{e}")

    def install_font_batch(self):
        """
        批量安装字体
        """
        if not os.path.exists(self.fonts_dir):
            raise FileNotFoundError(f"字体文件夹不存在：{self.fonts_dir}！")

        for font_file in os.listdir(self.fonts_dir):
            font_path = os.path.join(self.fonts_dir, font_file)
            if os.path.isfile(font_path) and font_path.lower().endswith(".ttf"):
                try:
                    # 检查字体是否已安装
                    if self.check_font_installed(font_path):
                        print(f"字体 {font_file} 已经安装，跳过安装过程！")
                    else:
                        print(f"正在安装字体 {font_file} 到 {self.destination_font_dir} ...")
                        self.install_font(font_path)
                        print(f"字体 {font_file} 安装成功！")
                except Exception as e:
                    print(f"字体 {font_file} 安装失败：{e}")

        self.refresh_font_cache()

class LoadCardImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path_1": ("STRING", {"default":''}),
                "image_path_2": ("STRING", {"default":''}),
                "image_path_3": ("STRING", {"default":''}),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("img1","img2","img3")
    FUNCTION = "load_card_img"
    OUTPUT_NODE = True
    CATEGORY = "HailuoTarot"
    DESCRIPTION = "tarot card deal"

    def load_card_img(self, image_path_1,image_path_2,image_path_3):

        tool=TarotDealCard()
        img, txt, name = tool.load_card_and_text(image_path_1)
        tar = tool.convert_to_target_format(img)
        imag1 = tar


        img, txt, name = tool.load_card_and_text(image_path_2)
        tar = tool.convert_to_target_format(img)
        imag2=tar


        img, txt, name = tool.load_card_and_text(image_path_3)
        tar = tool.convert_to_target_format(img)
        imag3= tar


        return (imag1, imag2, imag3)


