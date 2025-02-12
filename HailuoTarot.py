# Made by JimWang for ComfyUI
# 02/04/2023
import os
import torch
import random

from torchvision.transforms.functional import to_tensor, to_pil_image
from typing import List
from PIL import Image
import time
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
import random
from PIL import Image
from typing import List, Tuple

def load_image(drawn_cards: List[TarotCard]) -> Tuple[List[Image.Image], List[int]]:
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
            img = img.rotate(180)  # 将图像旋转 180 度
            flip_flags.append(1)  # 记录翻转标志为 1
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "card_num": ("INT", {"default":1}),
                "text": ("STRING", {"default":''}),
                "cur_round": ("STRING", {"default":'1'}),
                "all_flag": ("STRING", {"default": '0'}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "card1": ("STRING", {"default": '0'}),
                "card2": ("STRING", {"default": '0'}),
                "card3": ("STRING", {"default": '0'}),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("Card_NO_1","Card_NO_2","Card_NO_3","CARD_DESC_1","CARD_DESC_2","CARD_DESC_3","cur_round","all_card_name")
    FUNCTION = "card_deal"
    OUTPUT_NODE = True
    CATEGORY = "HailuoTarot"
    DESCRIPTION = "tarot card deal"

    def card_deal(self, card_num: int,image1,image2,image3, card1,card2,card3, text='', cur_round="1", all_flag="0"):
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

        # 初始化返回的图像和文本
        result = {
            "images": [self.get_blank_img() for _ in range(3)],
            "texts": ["" for _ in range(3)]
        }
        result["images"][0] = image1
        result["images"][1] = image2
        result["images"][2] = image3
        result["texts"][0] = card1
        result["texts"][1] = card2
        result["texts"][2] = card3

        print(f"current card1:{result["texts"][0]}")
        print(f"current card2:{result["texts"][1]}")
        print(f"current card3:{result["texts"][2]}")
        if cur_round==1:
            img, txt = self.load_card_and_text()
            tar = self.convert_to_target_format(img)
            result["images"][0]=tar
            result["texts"][0]=txt+","
            all_card_name=all_card_name+txt+","
        if cur_round==2:
            img, txt = self.load_card_and_text()
            tar = self.convert_to_target_format(img)
            result["images"][1]=tar
            result["texts"][1]=txt
            all_card_name = all_card_name + txt + ","
        if cur_round == 3:
            img, txt = self.load_card_and_text()
            tar = self.convert_to_target_format(img)
            result["images"][2] = tar
            result["texts"][2]= txt
            all_card_name = all_card_name + txt + ","
        # 循环执行塔罗牌相关作业
        if all_flag==1:
            for i in range(cur_round):
            # 抽取塔罗牌对象
            # 转换为目标格式
                img,txt=self.load_card_and_text()
                tar = self.convert_to_target_format(img)
            # 更新结果
                result["images"][i] = tar
                result["texts"][i] = txt
                all_card_name=all_card_name+txt+","

        # 返回结果
        return (result["images"][0], result["images"][1], result["images"][2], \
            result["texts"][0], result["texts"][1], result["texts"][2],cur_round,all_card_name)

    def load_card_and_text(self):

        drawn_cards = self.load_tarotCard(1)
        images, flags = load_image(drawn_cards)
        txt= load_text(drawn_cards)
        # 合并塔罗牌图片
        img = image_stitch(images)

        if str(flags[0]) == "1":
            txt=txt + "(Reversed)"

        return img,txt

    # 随机读取塔罗牌对象列表
    def load_tarotCard(self, card_nums: int) -> List[TarotCard]:
        tarotDeck = TarotDeck()
        tarotDeck.shuffle()
        if card_nums <= 0:
            card_nums=1
        drawn_cards = tarotDeck.draw(card_nums)
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




