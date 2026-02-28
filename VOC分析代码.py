import pandas as pd
import numpy as np
import os, re
import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# --- 必须添加的内容：资源初始化 ---
def init_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            # 检查资源是否存在
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            # 不存在则下载
            nltk.download(res)

init_nltk_resources()

# --- 1. 核心词库配置 (Feature Keywords) ---
FEATURE_DIC = {
        '颜色种类': {
            '正面-色彩丰富': ['many colors', 'lot of colors', 'plenty of colors', 'good range', 'great variety', 'great selection', 'every color', 'all the colors', 'so many options', 'variety of color is great', 'color assortment was ideal', 'nice assortment of colors'],
            '负面-色彩单调/反馈': ['limited range', 'not enough colors', 'wish for more', 'missing colors', 'disappointed with selection', 'needs more colors', 'wish that there were more colors'],
            '正面-套装/数量选择满意': ['love the large set', 'great number of colors', 'perfect amount of colors', 'huge set of 72', 'full set is amazing', 'good assortment', 'bought all sizes'],
            '负面-套装/数量选择不满意': ['wish for a smaller set', 'too many colors', 'no smaller option', 'forced to buy the large set', 'have to buy the whole set'],
            '正面-色系规划满意': ['great color selection', 'perfect pastel set', 'good range of skin tones', 'well-curated palette', 'love the color story', 'beautiful assortment of colors', 'has every color I need'],
            '负面-色系规划不满': ['duplicate colors', 'missing key colors', 'no true red', 'needs more grays', 'too many similar colors', 'palette is not useful', 'wish it had more pastels', 'poor color selection', 'needs more skin tones', 'pink needs to be more flamingo-ish', 'missing greens', 'needs more warm tones', 'repeated shades', 'lack of blue range', 'color was just like the orange'],
            '中性-提及色彩丰富度': ['color range', 'color variety', 'color selection', 'number of colors', 'range of shades', 'selection of hues', 'color assortment', 'color palette', 'spectrum of colors', 'array of colors', 'how many colors', 'color choice'],
       },
        '色彩一致性': {
            '正面-颜色准确': ['colors are exactly as expected','colors are just as you see','Perfect color match for what I needed','Colors were vibrant and just what I expected','stay true to their color','as shown in the photo', 'true to color', 'match the cap', 'accurate color', 'color accuracy', 'exact color', 'matches perfectly', 'consistent color', 'consistency', 'match the cap color', 'colors as expected'],
            '负面-颜色偏差': ['silver is a faded grey','gold appeared brown','white is not white','green is way too light','the greens have a lot of blue in them','Colors not quite accurate','the color is not as expected','colors are slightly different','color not what I needed','white not opaque enough','some colors swatch darker','some colors swatch lighter and more sheer',"red markers AREN'T RED","pastels don't show up well","color doesn\'t match", "shade doesn\'t match", 'misleading cap color', 'wrong cap color', 'color is off', 'darker than cap', 'lighter than cap', 'wrong color', 'color different', 'shade is different', 'red looks pink', 'green is lighter', 'orange looks like peach', 'gold has a greenish tone', 'darker than expected', 'colors are not the same', 'not the exact same','red looks more like coral','not as bright as I wanted'],
            '负面-两头颜色不一': ['slightly different color from one end to the other',"color doesn’t match between wide tip and fine point",'thin side vibrant, thick side much lighter','colors different from one end to the other','one end darker, one end lighter','thin side and thick side have different shades','smaller ends are a bit different color from the bigger side','larger nub color is not the same as the fine point','the smaller pen tip has slightly lighter ink than the bigger tip','thin side and thick side have different shades','When switching between tips the color varies a little','colors are different from one end to next', 'one end shows up darker', 'brush end was a totally different color', "ends don\'t match", 'colors actually clash with each other between tips'],
            '正面-设计-颜色准确 (VS 笔帽)': ['cap color matches ink color','colors are true to what you see on the cap', 'true to cap', 'true to color', 'match the cap', 'matches the cap perfectly', 'cap is a perfect match', 'cap is accurate', 'actually match the cap color','colors matched the cap colors',"matches what’s on the cap"],
            '负面-设计-颜色误导 (VS 笔帽)': ['color not as expected from the cap', 'lids are not the same color as ink','does not come with the colors in the picture','cap indicates red but ink is coral/pink',"paint isn’t true to color based on the pen caps",'None of them match the cap','caps are different colors but paint is the same','different colored caps but same ink', 'nowhere near the color on the outside of the pen','cap is a hint', 'cap is misleading', 'not even close to the cap', 'cap color is off', 'different from the lid', 'misleading cap', 'cap is wrong', 'cap is a lie', "color doesn't match the barrel", 'the cap color is way off', 'nothing like the cap', 'compared the caps and many of the colors are not the same'],
            '正面-营销-颜色准确(VS 网图)': ['beautiful colors just like the pictures','matches the description', 'exactly as advertised', 'what you see is what you get', 'matches the online photo', 'true to the swatch', 'photo is accurate'],
            '负面-营销-图片误导 (VS 网图)': ['colors do not look like what is pictured','not as bright as pictured','looks different from the online swatch', 'not the color in the picture', 'misrepresented color', 'photo is misleading', 'swatch card is inaccurate'],
            '正面-生产-品控(VS 其他笔)': ['color is consistent throughout', 'consistent color', 'consistency', 'no variation between pens', 'reliable color', 'batch is consistent'],
            '负面-生产-品控偏差(VS 其他笔)': ["didn't work consistently", 'inconsistent batch', 'color varies from pen to pen', 'my new pen is a different shade', 'no quality control', 'batch variation', 'def. different'],
            '中性-提及笔帽颜色': ['cap color', 'barrel color', 'match the cap', 'color of the cap', 'color on the barrel', 'cap match', 'indicator on the cap', 'swatch on the barrel', 'color indicated on the pen'],
            '中性-提及网图/色卡': ['swatch card', 'color swatch', 'online swatch', 'swatching', 'swatch test', 'product photo', 'online photo', 'listing photo', 'website image', 'advertised picture', 'photo in the listing'],
        },
        '色彩饱和度与混合': {
            '正面-鲜艳/饱和': ['bright colors', 'nice and bright', 'beautifully bright', 'richly saturated', 'perfectly saturated', 'deeply saturated', 'nice saturation', 'vibrant colors', 'rich colors', 'colors pop', 'brilliance of the colors', 'vivid color', 'very vivid', 'vividness', 'saturated color', 'bold colors', 'crisp and bright', 'vibrant and sharp', 'vibrant and wonderful color'],
            '负面-太鲜艳/刺眼': ['garish colors', 'colors are too loud', 'too neon', 'too bright', 'too fluorescent', 'overly bright'],
            '负面-暗淡/褪色': ['lackluster', 'muddy and washed-out', 'bit of dullness', 'faded a lil bit','dull', 'faded', 'pale', 'washed out', 'not bright', 'too pale', 'lackluster', 'colors are too dull', 'muddy colors', 'colors look dirty', 'desaturated', 'doesn\'t show up well', 'white is barely white', 'faint on the rock', 'will fade in the sun'],
            '正面-遮盖力强/不透明': ['great coverage', 'completely opaque', 'one coat covers all', 'thick paint feel', 'works on rocks', 'shows up on black', 'vibrant on dark surfaces', 'solid color', 'creamy texture', 'how opaque they are is awesome', 'solid shapes', 'cover in one coat', 'opaque and rich white', 'covers well when applied over other colors', 'pigmentation of the colors'],
            '负面-遮盖力差/透明': ['too transparent', 'very sheer', 'watery', 'streaky', 'takes many coats', 'see through', 'thin ink', 'runny', 'faint colors', 'watered down', 'req either 2 03 coats', 'requires a few coats', 'paint to be extremely thin', 'barely visible', 'could never solidly cover'],
            '正面-叠色/混合顺滑': ['layers well', 'easy to layer', 'good layering', 'smooth layering', 'layers perfectly', 'buildable color', 'smooth finish', 'not streaky', 'even layers', 'consistent texture', 'creamy application', 'doesn\'t disturb bottom layer', 'sits on top nicely', 'doesn\'t lift previous paint', 'no bleeding between layers', 'wet colors can be mixed', 'add water to the paper and blend', 'draw on top of the dried paint'],
            '负面-叠色/混合困难': ['takes forever to dry before layering', 'smears the bottom color', 'lifts if not 100% dry', 'lifts the layer underneath', 'scratches the paint off', 'rubs off previous layer', 'strips the paint', 'disturbs dried paint', 'tears up the bottom color', 'too transparent', 'very sheer', 'streaky when layering', 'can see the color underneath', 'takes too many coats to cover', 'patchy coverage', 'not so good for blending'],
            '中性-提及饱和度/混合': ['saturation', 'vibrancy', 'color intensity', 'richness of color', 'color depth', 'pigment load', 'high saturation', 'low saturation', 'deep saturation', 'blending'],
        },   
        '色系评价': {
    '正面-喜欢标准/基础色系': [
        'good standard colors', 'love the basic set', 'has all the primary colors', 'classic colors', 
        'great essential colors', 'perfect starter palette', 'all the fundamental colors', 
        'primary and secondary colors', 'standard rainbow colors', 'perfect basic selection'
    ],
    '正面-喜欢鲜艳/高饱和': [
        "colors come out vivid", "rich in colors", "bright and visible", "beautiful bright vivid colors", 
        "great color saturation", "beautiful pigmentation", "pigment is perfect", "rich and vibrant", 
        "highly pigmented", "gorgeous vivid hues", "stunning color output", "fantastic color payoff",
        "pigment is true to color", "beautiful bright colours", "clear and bright colors", 
        "bright colourful colors", "very vivid and saturated color", "vibrant and wonderful color", 
        "love the vibrant colors", "love the rich colors", "love the bold colors", "highly saturated", 
        "colors pop", "really pop", "makes the colors pop", "colors really stand out", "so full of life"
    ],
    '正面-喜欢粉彩色/柔和系': [
        "gentle color tones", "soft pastel shades", 'pastel colours are beautiful', 'love the pastel colors', 
        'beautiful pastels', 'adore the soft colors', 'perfect muted tones', 'calming color palette', 
        'soothing shades', 'subtle and elegant', 'love the macaron colors', 'macaron colors', 'pretty candy colors'
    ],
    '正面-喜欢复古/怀旧色系': [
        'love the vintage colors', 'perfect retro palette', 'nostalgic color scheme', 'love the old school colors',
        'adore the vintage feel', 'love the retro vibe', 'beautiful antique colors', '70s color palette', 
        'mid-century modern colors', 'love the mustard yellow', 'love the avocado green', 'love the burnt orange'
    ],
    '正面-喜欢莫兰迪色系': [
        'love the morandi colors', 'adore the morandi palette', 'perfect morandi palette', 'beautifully dusty colors', 
        'love the grayish tones', 'muted and elegant', 'sophisticated colors', 'understated and beautiful',
        'looks so high-end', 'elegant color scheme', 'love the muted aesthetic', 'hazy colors'
    ],
    '正面-喜欢中性/肤色系': [
        'love the skin tones', 'great range of skin tones', 'perfect for portraits', 'realistic skin tones', 
        'beautiful flesh tones', 'wide variety of skin colors', 'perfect neutral palette', 'love the neutral colors', 
        'great selection of neutrals', 'beautiful earth tones', 'flesh colored markers'
    ],
    '正面-喜欢大地/自然色系': [
        'love the earth tones', 'adore the earthy palette', 'beautiful natural colors', 'gorgeous nature-inspired colors', 
        'love the botanical colors', 'perfect botanical palette', 'beautiful forest greens', 'love the sage green', 'terracotta shades'
    ],
    '正面-喜欢灰色系': [
        'love the gray scale', 'great set of cool grays', 'perfect warm grays', 'good neutral grays',
        'excellent grayscale palette', 'adore the range of grays', 'beautiful selection of grays',
        'perfect for monochromatic work', 'great for shadows and shading'
    ],
    '正面-喜欢霓虹/荧光/夜光': [
        "glow pretty bright under blacklight", "UV reactive", "glow in the dark", "react under UV", 
        "glow under blacklight", "fluorescent neon", "glow is vibrant", "great blacklight reactive paint", 
        "show up well under blacklight", 'love the neon colors', 'super bright neon', 'vibrant neon shades', 
        'perfect for blacklight art', 'UV art'
    ],
    '正面-喜欢金属/珠光/亮片系': [
        "love the metallic ones", "great metallic ones", 'love the metallic colors', 'great metallic effect', 
        'beautiful metallic sheen', 'shiny metal finish', 'gorgeous chrome finish', 'looks like real metal', 
        'love the pearlescent finish', 'beautiful shimmer', 'amazing liquid chrome effect', 'very reflective', 
        'stunning iridescent colors', 'love the lustre', 'metallic and glittery', 'too sparkly', 'glitter lover'
    ],

    # --- 负面评价：侧重偏差、缺失与不实 ---
    '负面-显色偏差/严重色差': [
        "hue is off", "color is off", "not match cap color", "cap color not accurate", "red is very orange-red", 
        "pink is actually light pink", "red is orange", "green leans to teal", "colors dont look like the color it’s supposed to be", 
        "red is a dark pink", "the green is more like turquoise", "red is very pale like pink", 
        "only orange ones that are clearly not red", "the red that I have looks bright pink", 
        "light blue is more greenish", "color of pen doesn't show unless I do a second layer", 
        "looked rather washed out", "the white kept being picked up", "started to turn gray instead of white", 
        "not as vibrant as I thought", "not as bold as I hoped", "white is barely white", "red was definitely more on the pink side",
        "red look more like coral", "color not what I needed", "barely shows up", "hue doesn't match"
    ],
    '负面-效果名不副实(闪粉/金属)': [
        "vaguely coloured shiny mud", "little shinny but not glittery at all", "definitely not glitter", 
        "doesn't sparkle", "the glitter doesn't come across very well", "no actual glitter", 
        "no glitter or shimmer", "zero glitter whatsoever", "no sparkle at all", "not super metallic or shiney", 
        "just metallic", "lots of silver in the brush end", "DEFINITELY NOT GLITTER", "no real color just glitter", 
        "zero glitter to be found"
    ],
    '负面-色系搭配/质量评价': [
        "no real brown", "no deep colors", "triplicates of some colors", "no greens", "no true red", 
        "no standard red", "no blue", "no purples", "limited dark shades", "poor variety of colors", 
        "color range is uneven", "lots of pinks few greens", "color selection could be better", 
        "not enough browns", "too many purples", "too many yellows", "limited color range", 
        "duplicate light pink", "lack of darker shades", "odd color selection", "missing basic colors", 
        "only one bright red", "the color selection is not that great", "no standard blue or green", 
        "the only basic colors in this set are black and white", "no assortment of basics", 
        "there is no true brown or red", "color selection is a little bland", "missing critical shades", 
        "weird color combination", "terrible color choices", "bad color selection", "they kinda all look the same", 
        "no white included", "palette is ugly", "colors don't go well together", "poorly curated"
    ],

    # --- 中性描述：侧重客观提及 ---
    '中性-提及标准/基础色系': [
        'standard colors', 'basic set', 'primary colors', 'secondary colors', 'classic colors', 
        'essential colors', 'core colors', 'fundamental palette', 'traditional colors'
    ],
    '中性-提及鲜艳/饱和色系': [
        'vibrant colors', 'bright colors', 'bold colors', 'rich colors', 'vivid colors', 
        'color intensity', 'intense colors', 'highly saturated', 'brilliant colors'
    ],
    '中性-提及粉彩色/柔和系': [
        'pastel', 'pastels', 'pastel colors', 'soft colors', 'subtle shades', 'macaron colors', 
        'muted tones', 'muted colors', 'pale palette', 'baby pink', 'baby blue'
    ],
    '中性-提及复古/怀旧色系': [
        'vintage colors', 'retro palette', 'nostalgic colors', 'old school colors', 'sepia tones', 
        'antique colors', 'heritage colors'
    ],
    '中性-提及莫兰迪色系': [
        'morandi', 'morandi colors', 'morandi palette', 'grayish tones', 'muted palette', 
        'understated colors', 'dusty pink', 'sage green'
    ],
    '中性-提及中性/肤色系': [
        'skin tones', 'flesh tones', 'skin tone palette', 'portrait palette', 'neutral colors', 'neutrals'
    ],
    '中性-提及大地/自然色系': [
        'earth tones', 'earthy palette', 'natural colors', 'nature-inspired colors', 'botanical colors', 
        'forest greens', 'olive green', 'ocean blues', 'sky blue'
    ],
    '中性-提及灰色系': [
        'gray scale', 'grayscale', 'grays', 'shades of gray', 'cool grays', 'warm grays', 'charcoal gray'
    ],
    '中性-提及霓虹/荧光色系': [
        'neon colors', 'neon palette', 'fluorescent colors', 'highlighter colors', 'blacklight reactive'
    ],
    '中性-提及金属/珠光色系': [
        'metallic ink', 'metallic colors', 'metallic finish', 'liquid chrome', 'pearlescent finish', 
        'shimmer effect', 'glittering effect', 'iridescent', 'gold ink', 'silver ink'
    ],
    '中性-提及色系搭配': [
        'color palette', 'color combination', 'color scheme', 'color assortment', 'curated palette', 
        'range of colors', 'selection of colors'
    ],
},
        '笔头表现': {
            '正面-双头设计': ['love the dual tip', 'love two tips', 'convenient dual', 'brush and fine combo', 'best of both worlds'],
            '负面-双头设计': ['useless dual', 'redundant tip', 'unnecessary side', 'wish it was single', 'only use one side'],
            '正面-软头/笔尖好': ['love the brush', 'great brush', 'responsive nib', 'flexible tip', 'smooth application'],
            '负面-笔头磨损分叉': ['tip frays', 'frayed', 'split nib', 'wore out', 'lost its point', 'clogged', 'nib broke'],
            '正面-细节控制好': ['precise fine', 'perfect for details', 'crisp line', 'intricate work', 'sharp chisel', 'sturdy bullet'],
            '负面-细节控制差': ['scratchy', 'too broad', 'too thick','lost its edge', 'skips', 'bent the tip'],
            '正面-按压泵吸顺畅': ['easy to prime', 'primed quickly', 'fast to start', 'instant flow', 'smooth pump', 'ready in seconds'],
            '负面-按压泵吸差/漏墨': ['hard to prime', 'impossible to start', 'pumping for 10 minutes', 'stuck valve', 'nib receded', 'ink gushed', 'massive blob', 'messy splattered'],
            '正面-弹性/可替换': ['flexible', 'bouncy', 'nice spring', 'replaceable nibs', 'can replace tips'],
            '负面-硬度/不可替换': ['too stiff', 'too soft', 'mushy', 'no replacement', "can't replace"],
            '中性-提及笔头': ['dual tip', 'brush tip', 'fine liner', 'chisel nib', 'bullet point', 'dot marker', 'line variation', '0.5mm'],
        },
        '笔头耐用性': {
            '正面-耐磨损/抗分叉': ["brush tips have held up to repeated use and have kept their shape","the fine tip maintains a sharp point even after heavy use","haven't clogged or gotten gunky on the tips","don't clog when left uncapped","tips are nice and thick", "very durable tips", "tips stay fit even tho I’m heavy handed", "don't break down", "tips work well", "brush tips do not flick paint everywhere", "Good brush tip didn’t fray or dry out.", "tips hold up to the texture", "never fray or clog", "doesn't fray", "no fraying", "resists fraying", "not splitting", "still intact", "no signs of wear", "holds up", "tips are durable", "tips held up well", "don't fray or wear out quickly", "tips don't sink in if you press hard", "sturdy tips", "good quality tips", "tips are perfect", "tips are true to the design", "tips don't wiggle around", "dense felt tips", "tips don’t fray like some", "tips are precise", "the tips scrape previous layers away"],
            '负面-磨损/分叉': ["brush tip gets worn down very easily and smushes","the end of the “brush” tips are all “fuzzy”", "the tips of the markers shave off as you color leaving specs behind", "some of my nibs started to wear", "nibs are going to be wearing through that fast", "tip gets wide with a lot of use", "felt tips are too soft and tend to fray quickly", "the brush tips fray pretty easily", "pieces of the marker tip comes off onto the paper", "the fine tip comes off on the pages", "tips fray and stick to the paint leaving clumps in your art", "tips do lose the sharpness after multiple uses", "tips eventually \"round out\" after heavy use", "felt on the fine tip sheds", "tips wear out really fast", "didn't last very long", "tips frayed when using for a short time", "the fine side of the marker are quite delicate, so the fibers of the tip don't separate", "tips get fuzzy so quickly", "chisel tips frayed", "fine tip side frays", "tips were dry and cracked", "fibers start to separate", "tips are separating/fraying", "tip gets scratched up", "tips pill", "tips get a bit hairy", "felt is coming off of the tip", "tip gets fuzzy", "wear down the tip", "tip becomes frayed", "fray", "fraying", "frayed tip", "split nib", "splitting", "wears out quickly", "wear down fast", "tip is gone", "tips were frayed after about 10 minutes", "slight pilling of tip material", "fine points tend to become more like brush tips", "had to trim off the parts that had frayed", "tips don't seem to hold up", "tips frayed easily", "tips flatten out and/or split rather easily", "tips shred constantly", "tips fall apart quickly", "tips come off on the paper", "felt tip sheds", "doesn't take long to lose fine tip", "tips wear out quickly", "felt tips broke quickly", "the fine tip starts to 'shed' after a few uses", "slightly frayed tips on the fine side", "tips feathered out easily", "Product sheds too much", "every single tip sheds", "tips are fragile and split under the slightest pressure", "a few tips frayed with minimal use", "Overuse of brush tip will wear it out"],
            '正面-保形/硬度佳': ["both tips are firm and hold their shape well","tips are strong and don't bend","brush tips are keeping their shape", "flexible brush tips", "the tip is firm/thick", "brush tip is flexible", "fine tip has a hard tip", "tips are well made and clean cut", "tip acts just as it should, you can make a tiny line or press down a bit more to get a thicker line", "tip is very solid", "tip doesn’t have any give", "smoosh proof", "great tough tip", "hold its shape", "soft but firm enough to hold its shape", "tips hold their shape", "retains shape", "holds its point", "stays sharp", "doesn't get mushy", "doesn't go flat", "springs back", "good snap", "tip stays pointed", "smooth and stable the tips", "strong tips", "hold up well without turning to mush", "tips are solid", "not flimsy", "Has a hard tip I like", "the tip is pretty firm", "Solid fine tip", "maintain their shape", "tips don't get squashed out of shape in the caps"],
            '负面-形变/软化': ["tip is too soft","the fine tip is sort of scratchy", "tips get mushy so quickly", "super hard fine tip rips paper too easily", "black tip was bending loosen shape", "started loosing pigment the pen point started bending not keeping there shape", "Narrow tips bent and curled with no pressure applied", "lose their definition", "lose its shape", "become more fat", "become blunt", "tip gets smooshed", "bent tip", "gets mushy", "too soft", "tip softened", "spongy", "lost its point", "point went dull", "deformed", "went flat", "small tips got smashed easily", "fine side turned thick", "tip of the marker goes down", "tip becomes blunt", "deformed tip", "the tips sank into the barrel without a lot of pressure on them", "tips get squashed out of shape in the caps", "some of the wider 'dot' pen tips are deformed", "deformed tips", "tips are soft and fragile and not capable of keeping their shape", "the sharp point tip has no strength and after one use bends", "the tips get a little more wobbly", "the sharp point tip has no strength"],
            '正面-坚固抗损': ["Nice durable", "felt tips are very versatile", "tip quality is great", "tips have held up very well", "tips feel like they are good quality that will last a while","tips hold up beautifully", "tips are strong", "don't dull on the rocks", "good quality tip material", "tough nib", "durable tip", "sturdy nib", "robust", "heavy duty", "resilient", "withstands pressure", "doesn't break", "pens and tips are sturdy", "held up to child use", "durable tips", "high quality tips", "robust tips", "not fragile", "tips are super resistant to highlighting", "very durable", "tips are very durable", "sturdy design", "they’re very durable", "The markers feel sturdy", "resistant tip"],
            '负面-意外损坏': ["the tips/nibs of these markers come off", "tips fly off when I open them", "tip of the marker falls out", "if you have it dry and hold it tip down it will fall out", "couple others were missing the tip in the pen", "ruined tip for the fine side","tips push into tube with very little pressure", "tips are dull and one disappeared, retracted into the pen", "pen tip retreated into the marker", "tips push in all the way", "tip broke inside the tube", "tips of the markers falling out, popping out and breaking", "tips fall out on first use", "tip occasionally falls out", "tips have popped completely out", "ruin the tip within a couple uses", "tips continue to come off while painting", "tip very fragile", "tip broke off instantly", "every tip was a couple inches out from the marker with no way of pushing it back in", "tip falls out during use", "paint marker tip will fall out", "tip keeps falling out", "tip of the marker immediately breaks off", "tips are super long", "tips fall off a lot", "tips are too small for the pen and they keep falling", "tips were missing", "fine tips were broken", "tips just keep falling off", "fine tips are horrible and very flimsy", "tips are flimsy and wobble", "missing a chuck out of the tip", "tips were torn out", "small marker tips broke off", "tips stuck in the cap", "tips pulled out and broke", "tip popped off with the lid", "missing a tip", "tip broke", "fragile tips", "tips are too delicate", "the tip fell out", "half of the fine tips were dried out", "some of the tips were damaged", "one with the tip pushed in", "tips were pushed up from pressing down too hard", "the tip collapsed into into the pen", "The tip pushes in for the dot but is now stuck", "tips break easily", "tips are fragile", "tips are very fragile", "the tips covers have ridges which make it VERY difficult", "about 1/3 of these markers didnt work when they arrived", "3 of the fine tipped and 2 dot tipped pens were dried out", "Half of the markers were already dry with funky tips", "Not the best. They’re dry after a few uses", "the tip fell out on first use", "non working tip", "tips get stuck inside the marker", "tips are very flimsy"],
            '负面-寿命不匹配': ["dry out", "dried up", "dried out really fast", "dried out upon opening", "short lifespan", "quick drying out", "died quickly", "dries out too fast", "stop working","dried out", "off white pen dried up on the fine tip side", "fine tip dried out after one use","tips didn't last", "tip wore out before ink", "died before the ink", "ink left but tip is useless", "nib is gone but still has ink", "fine tip dried out while thick tip still works", "tips dried out before ink ran out", "the color ran out very quick especially on the dot tip side"],
            '正面-寿命长': ["don't dry out easily", "haven't dried up", "still write like they're brand new", "still going strong after", "Seem to be lasting well", "they haven't dried up and still write like they're brand new", "they will last for a while", "they last a good bit so far", "markers seemed to last forever", "pens -don't- dry out quickly", "these are long lasting","tips last a long time", "tips don’t fall apart as fast as others", "lasts a really long time", "tips last longer", "long lasting tips", "long lasting tip", "outlasts the ink", "good longevity", "lasts a long time", "lasted a long time with a lot of use", "longevity out of several brands", "still going strong after months", "long lasting", "still going after two weeks", "they last", "lasts really well", "so far they're lasting", "lasts for a long time", "Very impressed with the tips", "they last me awhile"],
        },
        '流畅性': {
            '正面-书写流畅': ["go on smoothly", "smooth writing", "smoothly glide on canvas", "smooth flow", "smoothly and effortlessly", "write smoothly", "apply smoothly", "paint goes on smoothly", "smooth pigment", "flows well", "continuous smooth flow", "smooth and even coverage", "flow smoothly", "glide effortlessly", "smooth paint", "smoothness of it is amazing","creamy, velvety, and smooth", "paint comes out smoothly and neatly", "smooth acrylic paint", 
                             "lay down a very smooth, even line", "colors flow in an even manner", "clean smooth lines","smooth flow from the tips", "very smooth to write with", "thick paint like ink","color doesn’t run or bleed", "smooth color application", "evenly distributed", "brush tip glides well", "smooth flow of the brush head", "looks smooth and vibrant", "smooth strokes", "flow of the paint is smooth and consistent", "smooth as silk", 
                             "ink flow is flawless and consistent", "smooth application", "paint flow is very nice","flowed super smooth", "write as easily and smoothly as a regular marker", "writes smooth", "smoothest markers", "ink flows at just the right amount", "tips and the paint run smooth","smooth and fluid", "flow smoothly without smudging", "smooth, even color", 
                             "ink flows smoothly", "silky smooth", "smooth and consistent stream of ink","buttery smooth", "glides on surface", "effortless application", "flows like a dream","no resistance", "smooth as butter", "glides across the canvas", "seamless flow", "very fluid"],
            '负面-干涩/刮纸/断墨': ["streaky, inconsistent results","stop working very quickly","hard to keep pen going smoothly","acts dry","Flow is not constant","not be as smooth","dried out quick","don't release paint smoothly","NOT a consistent, smooth writing","pulling up little shreds of the paper","paper peeling","pigment clogs up",'scratchy', 'feels scratchy on paper', 'scratches the paper', 'scratchy nib','writes dry', 'arrived dried out', 'dried up quickly', 'pen is dry', 'ink seems dry',
                         'skips', 'skipping', 'skips constantly', 'ink skips', 'hard start', 'hard starts', 'stops and starts','inconsistent flow', 'uneven ink flow', 'ink flow is not consistent'],
            '负面-出墨过多/漏墨': ["pours out a big clump", "ink will overflow","cause lumps","splatter","tiny little splatters","leaked","ink leaked everywhere","gush paint out","excessive drippage","globs of paint","paint pooling","bleed through", 'blotchy', 'splotchy', 'leaves ink blots', 'ink blobs','too much ink', 'puts down too much ink', 'gushes ink', 'ink gushes out', 'too wet',
                        'feathers badly', 'bleeds everywhere','leaking ink', 'leaked all over', 'ink leaked everywhere', 'leaky pen', 'arrived leaking'],
            '正面-速干且平整': ["fast drying", "not messy", "dries quickly", "dries flat", "no streaks", "smooth finish", "not watery", "solid line", "vibrant coverage", "thick paint feel", "professional finish", "uniform texture", "velvety finish", "opaque delivery", "dries pretty fast", "not messy at all", "dries quickly and doesn't bleed or feather", "mess-free application", "dry pretty fast", "dries almost instantly", "quick-drying formula", "dries fast", "dries rather quickly", "dried very fast", "dries fairly quickly", "dries super fast", "odorlessand quick dry", "quick dry to touch", "dried very quick", "fastest drying times", "dry time is minimal", "dries extremely fast", "dried quickly", "dried nicely", "Quick dry", "dry quickly", "dries water proof", "dried extremely fast"],
            '负面-漆感差/稀薄/起泡': ["paint bubbles up","bubble on the side", "paint is thin",'watery ink', 'too thin', 'pigment and water separated', 'faint color flow', 'bubbles in paint', 'air bubbles', 'frothy ink', 'see-through lines', 
                        'weak pigment', 'diluted color', 'runny paint'],
        },
        '墨水特性': {
            '正面-干燥快/防涂抹': ['doesn\'t feather', 'mess-free', 'mess-free application', 'no smudging or smearing', 'no bleeding', 'doesn\'t bleed or feather', 'clean finish', 'no splatter', 'no leakage', 'no clogging', 'no skipping', 'no blobs', 'no blotch', 'no blotch in sight','dry true to color', 'dry fairly quickly', 'dry fast enough to layer','quick dry', 'dry so fast','fast dry','not smear','not bleed','no bleed', 'not smear or bleed','dries quickly', 'dries instantly', 'dries immediately', 'fast-drying ink','no smear', 'no smudge', 'zero smear', 'zero smudge', 'smear proof', 'smudge proof',
                        'smudge resistant', 'smear resistant', 'doesn\'t smear', 'doesn\'t smudge','good for lefties', 'perfect for left-handed', 'lefty friendly','can highlight over it', 'highlight without smearing'],
            '负面-干燥慢/易涂抹': ['smudge if your hand accidentally touches the ink', 'ink did not dry and it smeared', 'smears lightly if it gets wet', 'paint lifting up if you layer on top of it before fully dry', 'smudges for left handed people','paint remains wet', 'transfers to other surfaces', 'takes 20+ minutes to dry','smears easily', 'smudges easily', 'smears across the page', 'smudges when touched', 'takes forever to dry', 'long drying time', 'never fully dries', 'still wet after minutes', 'slow to dry',
                        'not for left-handed', 'not for lefties', 'smears for left-handers', 'gets ink on my hand','smears with highlighter', 'smudges when layering', 'ruined my work by smudging'],
            '正面-环保/安全/无味': ['scentless','water-based', 'non-toxic formula', 'safe for children', 'no harsh fumes', 'non-toxic acrylic','non-toxic', 'AP certified non-toxic', 'certified non-toxic', 'no harmful chemicals', 'acid-free', 'archival quality', 'archival ink', 'photo safe','safe for kids', 'kid-safe', 'child-safe', 'great for children',
                        'no smell', 'no odor', 'odorless', 'scent-free', 'low odor', 'no fumes', 'no harsh smell', 'doesn\'t smell bad', 'xylene-free'],
            '负面-气味难闻': ['throat irritated', 'strong odor', 'tight and irritated throat','bad smell', 'strong smell', 'chemical smell', 'toxic smell', 'horrible odor', 'awful scent','overpowering smell', 'overwhelming fumes', 'nauseating smell', 'smells terrible',
                     'stinks', 'reek', 'stench', 'acrid smell', 'plastic smell','gives me a headache', 'headache inducing', 'smell is too strong', 'lingering smell'],
            '正面-持久/防水': ['waterproof', 'fade-resistant', 'long-lasting', 'preserving its beauty for years to come', 'adheres well', 'permanence once dry', 'doesn\'t rub off', 'stays vibrant even after drying', 'paint stays on','permanent when dry', 'resists casual rubbing', 'durable on surfaces', 'long-lasting on projects','truly permanent', 'permanent bond', 'archival quality', 'archival ink', 'museum quality','is waterproof', 'water resistant', 'doesn\'t run with water', 'survives spills', 'water-fast',
                      'fade proof', 'fade resistant', 'lightfast', 'excellent lightfastness', 'uv resistant', 'doesn\'t fade over time'],
            '负面-易褪色/不防水': ['not waterproof', 'smudges with water', 'ink bleeds when wet', 'rubs off', 'will rub off', 'washes off easily', 'scratches off', 'not permanent', 'fades too fast', 'colors have faded','comes off easily', 'rubs off with light touch', 'washes off glass', 'not waterproof on plastic', 'scratches off phone cases', 'not permanent', 'isn\'t permanent', 'fades quickly', 'fades over time', 'colors have faded', 'not lightfast','not waterproof', 'isn\'t water resistant', 'washes away', 'runs with water', 'smears with water',
                        'ruined by a drop of water', 'ink bleeds when wet'],
            '正面-续航长': ['ink last longer', 'plenty of ink', 'ink lasts forever', 'haven\'t dried up yet', 'lasted a long time', 'still going strong', 'great longevity', 'long-lasting ink', 'lasted a year of continuous use', 'ink payoff remains good', 'long-lasting with daily use','lasts a long time', 'lasted for months', 'seems to last forever', 'plenty of ink', 'large ink capacity', 'still going strong', 'has a lot of ink', 'haven\'t run out yet',
                    'great longevity', 'long-lasting ink'],
            '负面-消耗快': ['dried out really fast', 'dried out within a month', 'ran out of ink quickly', 'ink runs out fast', 'dries out after few uses', 'ran out after one use', 'used up too fast', 'doesn\'t last long','dries out after few uses', 'ink runs out super fast', 'dries out in a day', 'almost devoid of ink', 'ink runs out after one use', 'dries out within a month','runs out quickly', 'ran out of ink fast', 'dries out too fast', 'died quickly','empty fast', 'used up too fast', 'ran dry very quickly', 'doesn\'t last long','run out of paint',
                     'ink runs out in a day', 'not much ink inside', 'low ink capacity', 'wish it held more ink'],
            '正面-金属效果好': ['metallic silver and gold were included', 'metallics are very nice and reflective', 'shiny metal finish', 'beautiful metallic sheen', 'strong metallic look', 'nice reflective', 'metallic shine','rich metallic shimmer', 'shimmers in light', 'intense metallic look','great metallic effect', 'nice metallic sheen', 'shiny metal finish', 'strong metallic look', 'looks like real metal', 'beautiful chrome finish', 'very reflective'],
            '负面-金属效果差': ['not very metallic', 'dull metallics', 'not shiny', 'no metallic effect', 'looks flat', 'weak sheen', 'not reflective', 'metallic colors are dull', 'not sparkly', 'zero glitter','gold/silver coverage disappointing', 'not very metallic', 'dull metallics','dull metallic', 'not shiny', 'no metallic effect', 'looks flat', 'weak sheen', 'not reflective'],
            '正面-闪光效果好': ['lots of glitter', 'beautiful shimmer', 'sparkly', 'glitter is vibrant', 'nice pearlescent effect', 'very glittery', 'good sparkle'],
            '负面-闪光效果差': ['not enough glitter', 'no shimmer', 'glitter falls off', 'dull sparkle', 'barely any glitter', 'messy glitter'],
            '正面-荧光/霓虹效果好': ['almost neon bright', 'super fluorescent','neon pops', 'very bright neon', 'glows under blacklight', 'super fluorescent', 'vibrant neon', 'glows nicely'],
            '负面-荧光/霓虹效果淡': ['neon is dull', 'not very bright', "doesn't glow", 'not a true neon color', 'disappointing neon'],
            '负面-荧光/霓虹效果过饱和': ['too neon', 'too bright', 'too fluorescent', 'too neon/bright'],
            '正面-变色效果好': ['love the color change', 'chameleon effect is stunning', 'shifts colors beautifully', 'works in the sun', 'heat sensitive works'],
            '负面-变色效果差': ["doesn't change color", 'color shift is weak', 'barely changes', 'no chameleon effect'],
            '正面-夜光效果好': ['glows brightly in the dark', 'long lasting glow', 'charges quickly', 'very luminous'],
            '负面-夜光效果差': ["doesn't glow", 'glow is weak', 'fades too fast', 'barely glows'],
            '正面-香味好闻': ['smells great', 'love the scent', 'nice fragrance', 'fun scents', 'smells like fruit'],
            '负面-香味难闻/太浓': ['smell is too strong', 'bad smell', "doesn't smell like anything", 'chemical smell', 'artificial scent'],
            '中性-提及干燥/涂抹': ['drying time', 'dry time', 'smudge proof', 'smear proof', 'for left-handed', 'for lefties'],
            '中性-提及气味/安全': ['odor', 'smell', 'fumes', 'scent', 'non-toxic', 'acid-free', 'archival quality', 'chemical smell', 'safe for kids'],
            '中性-提及持久/防水': ['waterproof', 'water resistance', 'fade proof', 'lightfastness', 'lightfast rating', 'permanent ink', 'archival ink'],
            '中性-提及续航': ['longevity', 'ink life', 'how long they last', 'runs out quickly', 'runs dry', 'ink capacity'],
            '中性-提及金属效果': ['metallic ink', 'metallic effect', 'metallic sheen', 'chrome finish', 'reflective properties'],
            '中性-提及闪光效果': ['glitter ink', 'shimmer effect', 'sparkle', 'pearlescent effect', 'glitter particles'],
            '中性-提及荧光/霓虹效果': ['neon ink', 'fluorescent colors', 'under blacklight', 'glowing effect'],
            '中性-提及变色效果': ['color changing', 'color shift', 'chameleon effect', 'heat sensitive'],
            '中性-提及夜光效果': ['glow in the dark', 'luminous ink', 'glowing properties'],
            '中性-提及香味': ['scented ink', 'scented markers', 'fruit scent', 'fragrance'],
            '中性-提及可擦除性': ['erasable ink', 'can be erased', 'erases cleanly', 'frixion ink'],
        },
        '笔身与易用性': {
            '正面-材质/做工好': ['durable body', 'sturdy', 'sturdy build', 'well-made', 'solid construction', 'solidly built','quality feel', 'feels premium', 'high quality materials', 'quality build', 'well put together','feels substantial', 'built to last', 'high-grade plastic', 'metal construction', 'feels expensive'],
            '负面-材质/做工差': ['feels cheap', 'flimsy', 'cheap plastic', 'thin plastic', 'brittle plastic', 'feels plasticky', 'poorly made', 'poor construction', 'badly made', 'low quality build', 'fell apart',
                       'cracked easily', 'developed a crack','break', 'broke easily', 'broke when dropped', 'snapped in half', 'easy to break','doesn\'t feel durable', 'not sturdy'],
            '正面-握持舒适': ['comfortable to hold', 'comfortable grip', 'ergonomic', 'ergonomic design', 'ergonomic shape', 'nice to hold', 'feels good in the hand', 'feels great in the hand', 'good grip', 'soft grip',
                      'well-balanced', 'perfect weight', 'nice balance', 'fits my hand perfectly', 'contours to my hand', 'doesn\'t cause fatigue', 'no hand cramps', 'can write for hours', 'can draw for hours', 'reduces hand strain'],
            '负面-握持不适': [ 'uncomfortable to hold', 'uncomfortable grip', 'awkward to hold', 'awkward shape','causes hand fatigue', 'tires my hand quickly', 'gives me hand cramps', 'hand cramps up', 'hurts my hand', 'digs into my hand', 'sharp edges', 'too thick', 'too thin', 'too wide', 'too narrow', 'slippery grip', 'hard to get a good grip', 'poorly balanced', 'too heavy', 'too light', 'weird balance'],
            '正面-笔帽体验好': ['cap posts well', 'posts securely', 'cap posts nicely', 'secure fit', 'cap fits snugly', 'airtight seal', 'seals well', 'tight seal', 'cap clicks shut', 'satisfying click', 'audible click', 'easy to open cap', 'easy to uncap', 'cap stays on', 'doesn\'t dry out', 'durable clip'],
            '负面-笔帽体验差': ['hard to open cap', 'cap is too tight', 'difficult to uncap', 'struggle to open','loose cap', 'cap falls off', "cap doesn't stay on", "doesn't seal properly", 'not airtight',
                       'pen dried out because of the cap', 'dries out quickly','cracked cap', 'cap broke', 'cap broke easily', 'brittle cap','cap won\'t post', 'doesn\'t post securely', 'cap is too loose to post','clip broke off', 'flimsy clip', 'weak clip'],
            '正面-易于使用': ['easy to use', 'simple to use', 'user-friendly', 'intuitive design', 'no learning curve', 'effortless to use', 'easy to handle', 'easy to control', 'good control',],
            '中性-提及材质/做工': ['pen body', 'body material', 'barrel material', 'build quality', 'construction', 'plastic body', 'metal body', 'wooden body', 'resin body', 'surface finish', 'pen finish'],
            '中性-提及握持': ['grip section', 'grip comfort', 'ergonomic grip', 'pen balance', 'pen shape', 'barrel diameter',  'how it feels in the hand'],
            '中性-提及笔帽': ['pen cap', 'pocket clip', 'posting the cap', 'cap seal', 'screw cap', 'snap cap','caps','cap'],
            '中性-提及易用性/便携': ['portability', 'easy to carry', 'travel case', 'pen roll', 'for on the go', 'pocket pen', 'travel friendly'],
        },
        '绘画表现': {
            '正面-线条表现好/可控': ['good control', 'controllable lines', 'great line variation', 'crisp lines', 'consistent lines', 'clean lines', 'no skipping', 'sharp lines', 'great for fine details'],
            '负面-线条表现差/难控': ['hard to control', 'inconsistent line', 'uncontrollable', 'not for details', 'wobbly lines', 'shaky lines', 'broken line'],
            '正面-覆盖力好/不透明': ['opaque', 'good coverage', 'covers well', 'one coat', 'hides underlying color', 'works on dark paper', 'great opacity'],
            '负面-过于透明/覆盖力差': ['not opaque', 'too sheer', "doesn't cover", 'needs multiple coats', 'see through'],
            '正面-涂色均匀': ['even application', 'smooth application', 'no streaks', 'self-leveling', 'consistent color', 'no streaking'],
            '负面-涂色不均': ['streak', 'streaky', 'streaking', 'leaves streaks', 'patchy', 'blotchy'],
            '正面-兼容铅笔': ['goes over pencil cleanly', "doesn't smudge graphite", 'erases pencil underneath', 'covers pencil lines well'],
            '负面-铅笔兼容性差': ['smears pencil lines', 'smudges graphite', 'lifts graphite', 'muddy with pencil', "doesn't cover pencil"],
            '正面-兼容勾线笔': ["doesn't smear fineliner", 'works with micron pens', 'layers over ink', 'copic-proof ink compatible', 'safe over ink'],
            '负面-勾线笔兼容性差': ['smears fineliner ink', 'reactivates ink', 'lifts the ink line', 'bleeding with ink lines', 'makes ink run'],
            '正面-兼容水彩/水粉': ['layers over watercolor', 'works well with gouache', 'can use for watercolor effects', "doesn't lift watercolor"],
            '负面-水彩/水粉兼容性差': ['lifts watercolor', 'muddy with gouache', 'reactivates paint underneath', 'smears watercolor'],
            '正面-兼容彩铅': ['layers well with colored pencils', 'good for marker and pencil', 'blends with pencil crayon', 'works over wax pencil'],
            '负面-彩铅兼容性差': ['waxy buildup with colored pencils', "doesn't layer over pencil crayon", 'reacts weirdly with other markers'],
            '正面-兼容丙烯马克笔': ['layers nicely over Posca', 'can draw on top of Posca', "doesn't lift the acrylic", 'good with acrylic markers', 'adheres well to paint'],
            '负面-不兼容丙烯马克笔': ['smears Posca paint', "doesn't stick to acrylic marker", 'lifts the underlying acrylic', 'scratches off the acrylic surface'],
            '中性-提及线条表现': ['line quality', 'line control', 'line variation', 'stroke consistency', 'stroke', 'fine details', 'detailed work'],
            '中性-提及覆盖力': ['opacity', 'coverage', 'sheer', 'transparency', 'opaque', 'single coat', 'coverage strength'],
            '中性-提及涂色均匀性': ['even application', 'smooth application', 'streaks', 'streaky', 'patchy', 'blotchy', 'self-leveling'],
            '中性-提及可再加墨': ['reactivate', 'reactivation', 'lift', 'lifting', 'movable ink', 're-wettable'],
            '中性-提及兼容铅笔': ['over pencil', 'with pencil',],
            '中性-提及兼容勾线笔': ['over ink', 'with ink', 'over fineliner', 'with fineliner', 'over micron', 'copic-proof'],
            '中性-提及兼容水彩/水粉': ['over watercolor', 'with watercolor', 'with gouache', 'on top of paint'],
            '中性-提及兼容彩铅': ['over colored pencils', 'with colored pencils', 'over pencil crayon', 'with wax pencil'],
            '中性-提及兼容丙烯马克笔': ['on top of acrylic', 'over acrylic', 'with acrylic markers', 'with posca', 'on paint marker'],
        },
        '场景表现': {
            '正面-适合大面积填色': ["ideal for filling in larger areas", "filling in with the broader tip", "bold, sweeping strokes with ease", "covers larger areas with ease", "consistent coverage without streaks", "quick to cover large areas", "fills bigger sections fast", "wide nib for large coverage", "great for coloring", "good for large areas", "fills spaces evenly", "no streaking in large blocks", "coloring book friendly", "smooth coverage"],
            '负面-不适合大面积填色': ['may require two coats for best opacity on slick surfaces', 'can run out faster with heavy use','lines were sketchy and discolored','curls paper in large areas', 'pills paper when filling big sections', 'paint piles up on large spots','streaky when coloring', 'pills the paper when coloring large sections', 'dries too fast for large areas', 'bad for filling large spaces', 'leaves marker lines', 'patchy on large areas','takes forever to fill', 'not for solid blocks'],
            '正面-适合漫画/动漫创作': ['ideal for character art','great for manga', 'perfect for comics', 'blends skin tones beautifully', 'works for anime style', 'good for cel shading', 'great for character art'],
            '负面-不适合漫画/动漫创作': ['hard to blend skin tones', "colors aren't right for manga", 'smears my line art', 'not good for comic art'],
            '正面-适合插画创作': ['adds highlights to artwork','elevate your art projects','great for illustration', 'professional illustration results', 'layers beautifully for art', 'vibrant illustrations', 'perfect for artists'],
            '负面-不适合插画创作': ['not for professional illustration', 'colors are not vibrant enough for art', 'muddy blends for illustration', 'hobby grade only', 'muddy blends'],
            '正面-适合着色书/填色': ['great for colouring details', 'smooth on coloring pages', 'no bleed through thin paper', 'covers lines well', 'ideal for colouring books','great for coloring books', 'perfect for adult coloring', 'coloring book friendly', 'no bleed in coloring book', "doesn't ghost on coloring pages", 'safe for single-sided books', 'fine tip is perfect for intricate designs', 'great for mandalas', 'gets into tiny spaces'],
            '负面-不适合着色书/填色': ['bleeds through regular paper', 'leaves brush marks', 'paint blots on coloring pages', 'tips fray on coloring paper','not for coloring books', 'ruined my coloring book', 'bleeds through every page', 'does not cover the light grey lines of the coloring book', 'start to eat into the paper','ghosting', 'blobs of paint on the page', 'lost sharpness after multiple uses'],
            '正面-适合书法/手写艺术': ['perfect for calligraphy', 'great for hand lettering', 'nice thick and thin strokes', 'good for upstrokes and downstrokes', 'flexible tip for lettering', 'rich black for calligraphy'],
            '负面-不适合书法/手写艺术': ['tip is too stiff for calligraphy', 'hard to control line variation', 'ink feathers during lettering', 'not good for brush lettering', 'ink is not dark enough for calligraphy'],
            '正面-适合手工艺/物品定制': ['excellent for wedding guest book', 'signing wooden signs', 'decorating book bags', 'polymer clay coloring', 'key holder diy','great for diy projects', 'perfect for customizing shoes', 'works on canvas bags', 'permanent on rocks and wood', 'good for crafting'],
            '负面-不适合手工艺/物品定制': ["wipes off from plastic", "not for outdoor use", "color fades on fabric", "doesn't work on sealed surfaces"],
            '正面-适合儿童/教学': ['easy for small hands', 'keeps kids occupied', 'great for school carnivals','great for kids', 'safe for children', 'non-toxic', 'washable ink', 'durable tip for heavy hands', 'bright colors for kids', 'good for classroom use'],
            '负面-不适合儿童/教学': ['strong smell not for kids', 'ink stains clothes', 'tip broke easily with pressure', 'cap is hard for a child to open'],
            '正面-适合刻字/细节': ['control for small details', 'detailing work','3M fine tip for precision','perfect for lettering', 'great for calligraphy', 'nice for writing greetings', 'fine tip for small details', 'beautiful for sentiments'],
            '负面-不适合刻字/细节': ['too thick for tiny details', 'skips during fine work', 'not precise enough for small areas','too thick for lettering', 'bleeds when writing', 'hard to do calligraphy', 'tip is too soft for precision'],
            '正面-多表面DIY': ['works on denim jackets', 'permanent on stained wood', 'decorate ceramic and glass art', 'ideal for rock painting and garden decorations','painting on eggs and ornaments','works on metal and plastic art','painting pumpkins','customizing shoes and onesies','perfect for rock painting', 'works great on wood', 'customizing sneakers', 'painting on glass', 'ceramic decorating', 'canvas art', 'outdoor decor', 'painting pumpkins', 'ornament decorating'],
            '负面-不能多表面DIY': ['doesn’t work at all on DIY chalkboard','not permanent on glass if not sealed','wipes off without a second coat on slick surfaces','scrapes off glass', 'not permanent on glass', 'easily wipe off from glass', 'not for outdoor rocks', 'faded on outdoor use', 'not waterproof', 'smudges when wet','wipes off if not sealed', 'does not stick to metal', 'hard to start on metal', 'not for metal shop', 'not for non-paper surfaces', 'wipes off after even a week of sitting', 'paint doesn\'t stay on anything'],
            '中性-提及大面积填色': ['coloring large areas', 'filling in spaces', 'large coverage', 'background coloring'],
            '中性-提及漫画/动漫创作': ['manga', 'comic art', 'anime art', 'line art', 'character art', 'cel shading'],
            '中性-提及插画创作': ['illustration', 'illustrating', 'artwork', 'for my illustrations'],
            '中性-提及着色书/填色': ['coloring book', 'coloring books', 'adult coloring', 'colouring book', 'mandala', 'mandalas', 'intricate designs', 'coloring pages', 'secret garden', 'johanna basford', 'color by number'],
            '中性-提及书法/手写艺术': ['calligraphy', 'hand lettering', 'lettering practice', 'upstrokes', 'downstrokes','typography'],
            '中性-提及手工艺/物品定制': ['diy project', 'craft project', 'crafting with', 'customizing shoes', 'on canvas bags', 'on rocks', 'on wood', 'on plastic', 'on sealed surfaces'],
            '中性-提及儿童/教学': [ 'for kids', 'for children', 'in the classroom', 'for my students', 'art class', 'school project'],
            '中性-提及刻字/细节': ['lettering for cards', 'writing greetings', 'writing sentiments', 'for small details', 'for fine details', 'detailed work'],

        },
        '表面/介质表现': {
            '正面-在纸张上表现好': ['works great on marker paper', 'smooth on bristol board', 'blends well on bleedproof paper', 'perfect for mixed media paper', 'fit for paper', 'good for paper'],
            '负面-在纸张上表现差': ['still bleeds through marker paper', 'feathers on hot press paper', 'destroys bristol surface', 'pills my cold press paper', 'mess up your paper'],
            '中性-提及纸张': ['on paper','marker paper', 'bristol board', 'bristol', 'watercolor paper', 'mixed media paper', 'bleedproof paper', 'hot press', 'cold press', 'sketch book','for paper','for papers'],
            '正面-在深色纸张上显色好': ['opaque on black paper', 'shows up well on dark paper', 'great coverage on kraft paper', 'vibrant on colored paper', 'pops on black', 'shows up beautifully', 'great on black cardstock'],
            '负面-在深色纸张上显色效果差': ['not opaque on black', 'disappears on dark paper', 'too transparent for colored paper', "doesn't show up", 'color looks dull on black'],
            '中性-提及深色纸张': ['black paper', 'dark paper', 'kraft paper', 'colored paper'],
            '正面-在树脂/环氧上表现好': ['works great on resin', 'perfect for resin projects', 'great on epoxy', 'adheres well to resin', 'good on epoxy resin', 'works on epoxy', 'resin art friendly','good for resin crafts', 'resin marker', 'epoxy marker'],
            '负面-在树脂/环氧上表现差': ['doesn\'t work on resin', 'not for resin', 'won\'t stick to resin', 'pools on resin', 'runs on epoxy', 'slides off resin', 'smears on resin','won\'t dry on resin', 'not for resin molds', 'doesn\'t adhere to resin',
                                        'color runs on resin', 'bleeds on epoxy', 'streaky on resin', 'paint pools on epoxy','doesn\'t stick to smooth surfaces', 'won\'t stay on resin', 'runs into puddles',
                                        'not suitable for resin', 'not good on resin', 'useless for resin', 'forms puddles'],
            '正面-在布料上效果好': ['great on fabric', 'permanent on t-shirt', 'holds up in the wash', 'vibrant on textile', 'perfect for customizing shoes', "doesn't feather on cotton", 'survived the wash', 'applies smoothly to canvas', 'flexible on fabric', 'heat sets perfectly', "doesn't stiffen the fabric"],
            '负面-在布料上效果差': ['bleeds on fabric', 'feathers on canvas', 'fades after washing', 'washes out', 'makes the fabric stiff', 'washed right out', 'faded after one wash', 'cracked on the fabric', 'cracks when fabric flexes'],
            '中性-提及布料': ['canvas','canvas mural','on fabric', 'on canvas', 'on t-shirt', 'on textile', 'on cotton', 'on denim', 'for fabric', 'fabric marker'],
            '正面-在木材上表现好': ['great on wood', 'vibrant color on wood', 'dries nicely on wood', 'perfect for wood crafts', 'sharp lines on wood', 'beautiful finish on wood', 'seals nicely', 'vibrant on unfinished wood'],
            '负面-在木材上表现差': ['bleeds into the wood grain', 'color looks dull on wood', 'uneven color on wood', 'smears on sealed wood', 'bleeds with the grain', 'raised the wood grain', 'makes the grain swell'],
            '中性-提及木材': ['on wood', 'for wood', 'writes on wood', 'draw on wood', 'wood grain', 'sealed wood', 'wood crafts', 'unfinished wood'],
            '正面-在石头上表现好': ['great for rock painting', 'vibrant on rocks', 'opaque on stone', 'smooth lines on rocks', 'durable on pebbles', 'covers rocks smoothly', 'perfect for rock art', 'adheres well to stone', 'weather resistant', 'dries quickly on rocks'],
            '负面-在石头上表现差': ['scratches off rocks', 'not opaque enough for stone', 'color is dull on rocks', 'clogs tip on rough stone', 'hard to draw on rocks', 'chips off easily', 'too watery for rocks', 'streaky'],
            '中性-提及石头': ['on rock', 'on rocks', 'on stone', 'on stones', 'on pebble', 'on pebbles', 'for rocks', 'for rock painting', 'rock painting'],
            '正面-在粘土上表现好': ['works on polymer clay', 'great on air dry clay', 'vibrant on clay', 'soaks in nicely on bisque', "doesn't react with sealant", 'adheres perfectly to clay', 'bakes well', 'color stays true after sealing'],
            '负面-在粘土上表现差': ["doesn't adhere to clay", 'smears on polymer clay', 'clogs tip on un-sanded clay', 'reactivates the clay', 'melts the clay surface', 'never fully cures on clay', 'smears easily on polymer clay', 'reacts with glaze'],
            '中性-提及粘土': ['on clay', 'on polymer clay', 'on air dry clay', 'on bisque', 'for clay'],
            '正面-在玻璃(Glass)上表现好': ['permanent on glass', 'smudge proof on glass', 'crisp lines on glass', 'adheres well to glass', 'opaque on glass', 'vibrant on glass', 'writes smoothly on glass', 'removable with windex'],
            '负面-在玻璃(Glass)上表现差': ['wipes off glass', 'smears on glass', 'scratches off glass', 'beads up on glass', 'streaky on glass', 'difficult to remove from glass'],
            '中性-提及玻璃(Glass)': ['on glass', 'for glass', 'writes on glass', 'glass art', 'stain glass'],
            '正面-在陶瓷(Ceramic)上表现好': ['permanent on ceramic', 'writes on mugs', 'decorating ceramic', 'dishwasher safe', 'vibrant on ceramic', 'bake to set', 'cures to a hard finish', 'perfect for customizing mugs', 'great on mugs'],
            '负面-在陶瓷(Ceramic)上表现差': ['never dries on ceramic', 'wipes off ceramic', 'smears on ceramic', 'not dishwasher safe', 'washes off mug', 'scratches off ceramic', 'fades after baking', 'comes right off in dishwasher'],
            '中性-提及陶瓷(Ceramic)': ['on ceramic', 'on mugs', 'on glazed surface', 'for ceramic', 'decorating ceramic'],
            '正面-在塑料(Plastic)上表现好': ['permanent on plastic', 'smudge proof on plastic', 'adheres to plastic', 'vibrant on plastic', 'bonds to plastic', 'dries instantly on plastic', 'great on plastic models'],
            '负面-在塑料(Plastic)上表现差': ['wipes off plastic', 'smears on plastic', "doesn't stick to plastic", 'never dries on plastic', 'rubs off plastic', 'eats the plastic', 'remains sticky on plastic', 'remains tacky'],
            '中性-提及塑料(Plastic)': ['on plastic', 'for plastic', 'writes on plastic', 'plastic models'],
            '正面-在金属(Metal)上表现好': ['adheres to metal', 'permanent on metal', "doesn't scratch off metal", 'clean lines on metal', 'opaque on metal', 'dries quickly on metal', 'marks metal clearly', 'great for metalwork', 'weather resistant on metal'],
            '负面-在金属(Metal)上表现差': ['scratches off metal', 'smears on metal', 'wipes off metal', 'flaked off', 'peeled off metal', 'corrodes metal', 'takes forever to dry on metal', 'rubs off easily', "doesn't adhere to aluminum"],
            '中性-提及金属(Metal)': ['on metal', 'on aluminum', 'for metal', 'marks on metal'],
            '正面-在墙面上表现好': ['great coverage on walls', 'opaque on painted surfaces', 'covers in one coat', 'permanent on drywall', 'durable for murals', 'weatherproof', 'smooth on walls', 'great for mural work', 'low-fume for indoor use'],
            '负面-在墙面上表现差': ['wipes off the wall', 'not for outdoor murals', 'too transparent for walls', 'streaky on walls', 'damaged my wall'],
            '中性-提及墙面': ['on the wall', 'on walls', 'for murals', 'graffiti', 'on drywall', 'on plaster', 'on painted wall'],
        },
        '外观与包装': {
            '正面-外观/设计美观': ['awesome design and color', 'cool design','ergonomic design','beautiful design', 'minimalist design', 'sleek design', 'clean design', 'well-designed','thoughtful design', 'love the design', 'love the look of', 'pleasing aesthetic', 'looks elegant', 'high-end look', 'modern look', 'looks professional', 'impressed with the design',"visual appeal is great","professionally designed", "love the transparent section" ],
            '负面-外观廉价/丑': ["pen design feels flimsy", "cheap plastic construction",'looks cheap', 'feels cheap', 'cheaply made', 'cheap appearance', 'low-end look', 'plasticky feel', 'flimsy appearance', 'looks like a toy', 'toy-like', 'looks like a child\'s toy','ugly design', 'unattractive design', 'clunky design', 'awkward look', 'poorly designed', 'gaudy colors', 'tacky design', 'looks dated', 'outdated design'],
            '正面-包装美观/保护好': ["tight compacted box", "nice design on the cover box", "nicely packaged", "pens fit securely in slots","each marker individually wrapped in plastic","sturdy storage box","formed tray inside","convenient storage case","sturdy and organized zipper case","very well packaged", "securely packed", "well protected", "arrived in great shape", "packaging was good", "beautiful packaging", "nice packaging", "lovely box", "great presentation", "well presented", "elegant packaging", "giftable", "perfect for a gift", "great gift box", "nice enough to gift", "well packaged", "packaged securely", "protective packaging", "excellent packaging", "sturdy case", "durable case", "high-quality box", "nice tin", "reusable case", "great storage tin", "comes in a nice case","packaged lovely", "packaged was secured",  "packaged great", "packaged very nicely", "individually wrapped and packaged well", "the box sealed, each marker is sealed in plastic"],
            '负面-包装廉价/易损坏': ["need to redesign packaging", "box could be designed better", "no sturdy storage box","flimsy trays inside","came loose without standard box","package bent out of shape","boxes smashed and ripped apart","ripped bag","packaging was a mess", "flimsy packaging", "cheap packaging", "thin cardboard", "poor quality box", "doesn't protect the pens", "damaged box", "crushed box", "dented tin", "arrived damaged", "damaged in transit", "damaged during shipping", "broken case", "cracked case", "case was broken", "clasp broke", "latch doesn't work", "zipper broke", "cheap case", "flimsy case",  "the pens didn’t come in a box, they came in a ripped bag", "arrived in a flimsy, dented box", "the box was basically spilling the markers", "opened and tampered", "came bundled in rubber bands and plastic bags", "no container so you will have to purchase one separately"],
            '负面-包装过度':["excessively packaged", "overly protected", "too much packaging", "excessive protection", "individually wrapped with plastic which seems wasteful", "too much plastic wrapping", "unnecessary layers of packaging", "hard to unwrap each pen", "packaging is excessive for the product"],
            '负面-标签与真伪争议 (针对海外版/进口版)': ['FAKE PRODUCT', 'NOT REAL DEAL', 'fake POSCA', 'Japanese under label is fake', 'logo different','no directions in English','labeling all in Japanese','everything imprint in Japanese','everything written in Chinese', 'all writing is Asian', 'Japanese import labels', 'no English or French on packaging', 'does not conform to packaging laws', 'writing on the sides came in Japanese', 'different from the pictures shown','looks like a knock off', 'off brand items'],
            '正面-收纳便利': ["easy to store", "horizontal storage", "reusable storage","easy to re-package","handy for carrying","take everywhere","easy to store horizontally","easy to store", "convenient for carrying", "pens stay in place", "organized neatly in the box", "easy to keep track of colors", "storage is convenient",'well-organized', 'keeps them neat', 'keeps them organized', 'easy to organize', 'easy access to colors', 'easy to find the color', 'easy to get pens out', 'convenient storage', 'handy case', 'sturdy case', 'nice carrying case', 'protective case', 'pens fit perfectly', 'individual slots for each pen', 'great storage box', 'useful pen holder'],
            '负面-收纳不便': ["hard to sort unlabeled markers","need to buy solid plastic package","need separate holder","no removable tray","pens don't stay in place","hard to get all back in box","no easy storage once out of box","hard to get out", "difficult to remove pens", "pens are too tight in the slots", "struggle to get them out","messy organization", "poorly organized", "pens fall out of place", "don\'t stay in their slots","no individual slots", "pens are all jumbled together", "hard to put pens back","case doesn\'t close", "case doesn\'t latch", "lid won\'t stay closed", "clasp broke", "zipper broke","flimsy trays", "pens fall out when opened"],
            '中性-提及外观': ['pen design', 'overall look', 'visual appeal', 'aesthetic', 'appearance', 'form factor', 'finish', 'color scheme'],
            '中性-提及包装': ['packaging', 'box', 'outer box', 'sleeve', 'tin case', 'gift box', 'presentation', 'protective case', 'blister pack', 'unboxing'],
            '中性-提及收纳': ['storage case', 'carrying case', 'pen holder', 'pen stand', 'pen roll', 'organizer tray', 'layout of the tray', 'how they are organized'],
        },
        '多样性与适配性': {
            '正面-用途广泛': ["great for customizing hats, shirts, or fabrics", "used on chalkboard, finished wood, and glass", "great for various hand lettered paint projects", "great for cards, posters", "perfect for doing calligraphy and graffiti style art on poster board", "great for customizing clothes, decorating stones or creating glass art", "perfect for both professional projects and homemade crafts", "ideal for rock painting with children", "great for beading designs, household decor, crafts, jewelry", "perfect for both beginners and those who enjoy art and crafts", "perfect for various projects", "perfect for any school projects", "excellent for all my painting projects", "works well on textiles, denim, vinyl", "works beautifully on mirror, wood and glass", "perfect for writing on mirror, wood and glass", "excellent for adding crisp, clean lines or fine details to craft projects", "perfect for outlining, lettering, and intricate designs", "great for mixed media work", "great for many different projects", "works on a variety of surfaces", "use on various surfaces", "multi-surface use", "use for a variety of projects", "perfect for all sorts of projects", "multi-purpose", "all-in-one", "jack of all trades", "works for everything", "use it for everything", "handles a variety of tasks", "works on multiple surfaces", "use on different surfaces", "good for many different projects", "one set for all my needs", "great for both drawing and writing"],
            '负面-用途单一': ["not ideal for specialized calligraphy", "only built for art paper/canvases", "not suitable for alternative surface use", "not for multi media exploration", "not meant for all surfaces", "limited use on certain materials", "not versatile", "lacks versatility", "not multi-purpose", "single-purpose", "single use", "one-trick pony", "limited use", "very limited in its use", "limited application", "only for paper", "only works on paper", "doesn't work on other surfaces", "only good for one thing", "useless for anything else", "very specific use"],
            '正面-可拓展性 (Collection can be expanded)': ["pastel palette available", "impressive color range", "diverse color selection", "expandable collection", "can add to my collection", "love adding to my collection", "complete my collection", "collect all the colors", "love that they release new sets", "new colors available", "hope they release more colors", "can't wait for new colors", "limited edition colors", "love the special editions", "collector's edition"],
            '负面-可拓展性差 (Poor expandability)': ["no variety in yellow tones", "no variety in purple tones", "limited color range", "colors are a bit limited", "lack of bold/true colors", "limited number of colors available", "only pastel shades", "not enough grey tones", "not enough traditional Christmas colors", "only a few dark and metallic colors", "no hundred other colors available", "wish they had more muted colors", "looking for a full range but not getting it", "no new colors", "collection is limited", "wish they had more shades", "no new sets released", "stagnant collection", "line seems to be discontinued", "never release new colors", "can't expand my collection", "no updates to the color range", "stuck with the same colors", "wish they would expand the range", "color range is too small", "no new releases"],
            '正面-可补充性 (Can be replenished)': ["various point tips available", "can purchase smaller sized markers", "available in different size sets", "backup pens recommended", "replacement set easily purchased", "buy additional sets for backup", "replacement tips included", "multiple backups available", "can be refilled", "refillable", "reusable", "refill", "refills", "refill paint", "easy to refill", "refilled", "paint refill", "refillable with", "recyclable", "tips are reversible", "refillable", "refillable ink", "ink refills available", "can buy refills", "replaceable cartridges", "buy individually", "can buy single pens", "sold individually", "available as singles", "open stock", "don't have to buy the whole set", "can just replace the one I need", "replaceable nibs", "can replace the nibs", "replacement nibs available"],
            "负面-可补充性差 (Poor replenishability)": ["no individual nib replacements", "tips are not replaceable", "tips are not refillable", "markers are not refillable", "cannot be refilled", "not refillable", "disposable", "one time use pen", "dried out and not refillable", "sealed and not refillable", "not replaceable", "not refillable", "replacement tips cost almost as much as a new marker", "difficult to find replacement tips", "dried out and not refillable", "can't buy single", "not sold individually", "not available individually", "can't buy individual pens", 
                                                     "not sold as singles", "wish they sold refills", "no refills available", "can't find refills", "ink is not refillable", "no refill cartridges", "no replacement nibs", "can't replace the tip", "no replacement parts", "have to buy a whole new set", "forced to rebuy the set", "must buy the entire set again"],
            "正面-单支购买": ["buy singles to replace those 3", "colors are available in singles to purchase", "available as singles", "can buy single white pens", "available as individual markers", "don't need to buy a whole pack for one color", "sold as singles for replacement"],
            "负面-不可单支购买": ["wish I could buy these markers easily individually", "can't buy individual colors", "forced to buy a set for one color", "wasteful to buy a new set", "no single replacements", "can't find individual pens for sale", "forced to buy 12 just for the black one"],
            '中性-提及用途广泛性': ['versatility', 'multi-purpose', 'all-in-one', 'works on multiple surfaces', 'use for different things', 'all purpose', 'various uses'],
            '中性-提及可拓展性': ['expandable collection', 'add to the collection', 'complete the set', 'new colors','new sets released', 'limited edition', 'collect all the colors'],
            '中性-提及可补充性': ['refillable', 'open stock', 'sold individually', 'buy single pens', 'replacement nibs', 'ink refills', 'refill cartridges'],
            
            },
        '教育与启发': {
            '正面-激发创意/乐趣': ['fun to use', 'so much fun to play with', 'a joy to use', 'enjoyable to use', 'very satisfying','inspires me to create', 'makes me want to draw', 'makes me want to create', 'sparks my creativity',
                        'boosts my creativity', 'unleashes creativity', 'creative juices are flowing','gets me out of a creative block', 'helps with creative block', 'opens up new possibilities','creativity is unlimited', 'make art easy and successful', 'incredible value for money', 'creative revolution in a box', 'imagination run wild', 'unleash your inner Picasso', 'unleash creativity', 'perfect for anyone looking to improve their drawing skills', 'game-changer for art projects', 'highly recommended for artists of all ages and skill levels', 'puts me in a state of flow and relaxation', 'creative win-win for family crafting nights'],
            '正面-适合初学者': ['great equalizer for the beginner artist','perfect for hobbyist', 'great for art educator', 'easy to use', 'great for kids', 'kids safe','beginner friendly', 'good for beginners', 'easy for a beginner', 'perfect for beginners','easy to start', 'great starting point', 'just starting out', 'getting started', 'starter kit', 'great starter set', 'my first set', 'new to art', 'new to painting', 'new to drawing', 'first time trying','easy to learn', 'easy to learn with', 'no learning curve', 'no prior experience needed','simple enough for kids to use', 'great for casual artist', 'perfect for those new to paint pens', 'great trial run for beginners', 'no experience needed to create nice artwork', 'ideal for both beginners and experienced crafters'],
            '负面-有学习门槛': ['steep learning curve', 'learning curve', 'not for beginners', 'not beginner friendly','hard to use', 'difficult to use', 'confusing to use', 'not intuitive', 'hard to control',
                       'difficult to get the hang of', 'takes a lot of practice', 'requires a lot of skill','frustrating for a beginner', 'not easy to get started with'],
            '正面-有教学支持': ['clear instructions','excellent instructions','super easy instructions','easy to follow instructions','in depth instructions','detailed instructions','helpful guide', 'clear instructions', 'easy to follow guide', 'step-by-step guide',  'well-written instructions', 'great instruction book','good tutorial', 'helpful video tutorial', 'easy to follow tutorial','great community', 'supportive community', 'helpful facebook group', 'comes with practice sheets', 'love the worksheets', 'great online course'],
            '负面-无教学支持': ['no info except in Japanese','no guidance','instructions not written in English','no English instructions','no instructions', 'no guide included', 'didn\'t come with instructions', 'no user manual', 'lacks instructions', 'confusing guide', 'unhelpful guide', 'hard to understand instructions', 'instructions are not clear', 'useless instructions', 'poorly written', 'vague instructions', 'bad translation','instructions in another language', 'only in chinese',
                       'no online tutorials', 'can\'t find any videos on how to use'],
            '中性-提及创意/乐趣': [ 'creative juices', 'fun activity', 'joy of creating', 'spark creativity', 'boost creativity','creative outlet', 'artistic expression', 'fun to use', 'enjoyable process', 'doodling for fun'],
            '中性-提及学习门槛': ['beginner friendly', 'good for beginners', 'easy for a beginner','starter kit', 'starter set', 'my first set', 'entry-level','learning curve', 'no prior experience', 'easy to learn with',
                        'just starting out', 'getting started','new to art', 'new to painting', 'new to drawing','learning to draw', 'learning to paint'],
            '中性-提及教学支持': ['instruction book', 'instructional booklet', 'guidebook', 'step-by-step guide', 'how-to guide', 'learning guide', 'video tutorial', 'youtube tutorial', 'following a tutorial',
                        'online course', 'skillshare class', 'practice sheets', 'worksheets','online community', 'facebook group'],
        },
        '特殊用途': {
            '正面-专业级表现': ["finish looks professional", "looks professional", 'professional grade', 'artist grade', 'pro grade', 'professional quality', 'artist quality', 'studio grade', 'museum quality', 'for serious artists', 'not student grade','professional results', 'gallery quality results', 'publication quality',
                       'industry standard', 'lightfast', 'excellent lightfastness', 'high lightfastness rating', 'fade-resistant', 'fade proof', 'archival quality', 'archival ink', 'archival pigment',"professional looking artwork", "professional paint markers", "look extremely professional", "professional-looking results", "publication quality", "long-lasting and archival-quality artwork", "archival-quality artwork"],
            '负面-非专业级': ["not professional grade","not for professional art", "not professional", "not professional grade", "not professional quality", "kids projects not professional", "any professional art", "best for kids projects", "great for hobbies and amateur use", "perfect for children and casual art", "for casual use", "not for serious artists", "am not a professional", "not a professional artist", "novice tasks", "young artists", "good starter set", "good for beginners"],
            '中性-提及专业性': ['professional grade', 'artist grade', 'hobby grade', 'student grade', 'pro grade', 'lightfast', 'lightfastness rating', 'archival quality', 'archival ink', 'museum quality'],
        },
        '性价比': {
            '正面-性价比高': ['great value for the money', 'good value for the money', 'well worth the money', 'value for money', 'great value for money', 'worth my money', 'good deal for money', 'great value for your money', 'worth the extra money', 'excellent value for the money', 'nice value for the money', 'so worth my money', 'pretty expensive but totally worth the money', 'worth the money and time', 'really good value for your money', 'great value and totally worth the money','great product great price', 'very great price', 'good price', 'great price', 'worth the price', 'well worth the price', 'nice price', 'very reasonable', 'best price', 'amazing price point', 'price is on point', 'reasonable price', 'price is very reasonable', 'unbeatable price', 'fantastic value', 'steal', 'worth every bit', 'price was perfect', 'very affordable', 'very reasonably priced', 'bargain', 'deal','affordable', 'cheap', 'great deal', 'worth the money', 'great buy', 'reasonable price', 'cheaper than', 'alternative to','excellent value', 'amazing value','inexpensive','low price', 'great price point','money well spent', 'can\'t beat the price'],
            '负面-价格昂贵': ['not worth the money', 'waste of money', 'total waste of money', 'complete waste of money', 'a waste of money', 'money waster', 'not worth your money', 'don\'t waste your money', 'waist of money', 'just a waste of money', 'not worth the time or money', 'sadly a huge waste of money', 'don\'t have money to waste on stuff that doesn\'t work', 'would be more with my time and money if I never got them at all','over priced', 'bit pricey', 'too pricey', 'quite pricey', 'little overpriced', 'premium price', 'higher price', 'pricey though', 'a little expensive', 'price is high', 'not worth the price','expensive', 'overpriced', 'price they charge it\'s not worth', 'pricey', 'costly', 'rip off', 'waste of money','not worth it','over-priced'],

        },
        '配套与服务(色卡)': {
            '正面-提供色卡/好用': ['helpful color reference','easy to reference the color chart','love the included color guide','clear color swatch card','useful color identification card','includes a color swatch booklet','comes with a color guide','came with a color reference card','love the swatch card', 'includes a color card', 'comes with a card to help identify each color', 'comes with a color card to fill in', 'handy card to see what the colors look like', 'came with a color card to fill in', 'comes with a card to swatch colors', 'easy to make a color card', 'comes with a swatch card', 'includes a swatch card', 'helpful swatch card', 'great for swatching', 'easy to swatch', 'blank swatch card', 'pre-printed swatch card'],
            '负面-缺少色卡/不好用': ['no color reference included','missing color identification labels','difficult to match the swatch card','color chart is not included','no visual color reference','color swatch is not provided','hard to identify colors without a chart','wish it had a color code/name on each marker', 'not easy to make a color card', 'no color code/name on each marker', 'color card is not accurate', 'color card wasn\'t especially true to color', 'markers are unlabeled', 'colors not numbered', 'no swatch card', 'wish it had a swatch card', 'doesn\'t come with a swatch card', 'had to make my own swatch card', 'swatch card is inaccurate', 'swatch card is useless', 'colors on swatch card don\'t match'],
            '中性-提及色卡': ['swatch card', 'color chart'],
        },
        '购买与服务体验': {
            '正面-开箱/展示/运输好': ['easy ordering', 'quick arrival time', 'speeded service','arrived in good condition','arrived in perfect condition','arrived in excellent condition','arrived perfectly packaged','arrived well packaged','arrived quickly','arrived early','arrived amazingly quick','arrived in record time','arrived within 1-2 days','arrived in timely manner','arrived promptly','arrived in great time','arrived very fast','arrived a day earlier than expected','arrived well packed','arrived on time','excellent presentation','fast shipping', 'prime shipping', 'quick delivery','beautiful presentation', 'great unboxing experience', 'perfect for a gift', 'looks professional', 'elegant packaging', 'giftable', 'nice gift box', 'well presented', 'impressive presentation',
                       'lovely box', 'makes a great gift', 'nicely laid out'],
            '负面-运输/损坏': ['already primed', 'without plastic covers','arrived without outer packaging','package was lost','arrived dry','arrived unusable','over a week late','took much longer','arrived almost a month','damaged on arrival', 'missing marker', 'pen was missing', 'not all included', 'empty slot', 'marker missing', 'one less than expected', 'incomplete set',
                      'arrived broken', 'arrived broken', 'pens arrived broken', 'some were broken', 'cracked on arrival', 'damaged during shipping','damaged in transit', 'arrived damaged', 'item was damaged','leaking ink', 'leaked all over', 'ink leaked everywhere', 'arrived leaking','box was crushed',
                      'package was damaged', 'box was open', 'dented tin', 'poorly packaged for shipping', 'not well protected', 'arrived in bad shape','didn’t receive all the colors', 'picture shows 15 and I received 8', 'only 8 markers', 'missing markers in set'],
            '正面-客服/售后': ['totally satisfied', 'stand behind their products', 'admirable', 'sorted out the problem', 'grateful for the refund', 'easy to get a refund', 'care about customer satisfaction', 'handled the situation immediately', 'remedy the problem','free replacement', 'reimbursement offered', 'helpful support', 'prompt service','great customer service', 'excellent customer service', 'amazing support', 'seller was helpful', 'seller was very helpful', 'very responsive seller', 'quick response', 'fast reply',
                      'answered my questions quickly', 'resolved my issue quickly', 'problem solved','fast replacement', 'quick replacement', 'sent a replacement right away', 'easy replacement process',
                       'easy refund', 'hassle-free refund', 'full refund was issued','went above and beyond', 'proactive customer service'],
            '负面-客服/售后': ['no resolution', 'ignored complaint','missed return date', 'difficult to return', 'past return window', 'no refund', 'deceptive description', 'act of deception','bad customer service', 'terrible customer service', 'poor support', 'no customer service','seller was unresponsive', 'no response from seller', 'never replied', 'took forever to respond', 'slow response',
                      'seller was unhelpful', 'refused to help', 'unwilling to help', 'could not resolve the issue','missing items', 'missing parts', 'didn\'t receive all items',
                       'wrong item sent', 'received the wrong color', 'sent the wrong size','difficult return process', 'hassle to get a refund', 'refused a refund', 'no replacement offered'],
            '中性-提及开箱/展示': ['unboxing experience', 'presentation', 'packaging', 'giftable', 'nice box', 'sturdy case', 'storage tin', 'well organized', 'comes in a case'],
            '中性-提及运输': ['shipping', 'delivery', 'arrival condition', 'transit', 'shipped', 'arrived',  'damage','damaged', 'broken', 'crushed', 'shipping box', 'protective packaging'],
            '中性-提及客服/售后': ['customer service', 'contacted seller', 'contacted support', 'seller response','replacement', 'refund', 'return process', 'exchange', 'missing items', 'wrong item sent', 'issue resolved'],
        },
        '容量': {
            '正面-容量大/耐用': ['large capacity', 'holds a lot of ink', 'plenty of ink', 'longevity', 'long lasting', 'not run out quickly'],
            '负面-容量小/消耗快': ['ran out of ink', 'running out of ink', 'low ink capacity', 'runs out fast', 'ink runs out too fast', 'short lifespan','died quickly', 'empty after']            
        }
}
# --- 2. 产品分类映射 (Product & Category Mapping) ---
USER_CATEGORY_MAPPING = {
    'B001AS6P4G': 'B001AS6P4G_posca_(>2.56)_阀门式_8mm_Chisel_15_纸盒_插口盒',
    'B00ZC9JK4Q': 'B00ZC9JK4Q_posca_(2.00-2.56)_阀门式_1.8-2.5mm_Medium_7_纸盒_插口盒',
    'B013NMZ92U': 'B013NMZ92U_posca_(2.00-2.56)_阀门式_0.9-1.3mm_Fine_16_PET插口盒_插口盒',
    'B06ZYTZXG2': 'B06ZYTZXG2_posca_(2.00-2.56)_阀门式_0.9-1.3mm_Fine_24_纸盒_天地盖硬纸盒',
    'B07D2LC8LH': 'B07D2LC8LH_ARTISTRO_(1.50-2.00)_阀门式_0.7mm_Extra Fine_5_纸盒_插口盒',
    'B07NRW6H8M': 'B07NRW6H8M_ARTISTRO_(1.50-2.00)_阀门式_0.7mm_Extra Fine_12_纸盒_插口盒',
    'B086XCN9CR': 'B086XCN9CR_AKARUED_(1.07-1.50)_阀门式_2-3mm_Medium_8_PET插口盒_插口盒',
    'B08LTYXQLC': 'B08LTYXQLC_ArtShip Design_(1.50-2.00)_阀门式_0.7-2mm_Extra Fine&Medium_14_纸盒_插口盒',
    'B098QPHCVT': 'B098QPHCVT_AKARUED_(1.07-1.50)_阀门式_0.7mm_Extra Fine_8_纸盒_插口盒',
    'B09SWJRSSX': 'B09SWJRSSX_JR.WHITE_(0.59-0.79)_阀门式_0.7mm_Extra Fine_18_纸盒_插口盒',
    'B09TR9SD6P': 'B09TR9SD6P_SAKEYR_(1.07-1.50)_阀门式_2-3mm_Medium_12_纸盒_插口盒',
    'B09VCYS41G': 'B09VCYS41G_Betem_(0.47-0.59)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0B7VZ9KTJ': 'B0B7VZ9KTJ_Artugn_(0.47-0.59)_棉芯式_0.5-5mm_Brush&Fine_24_纸盒_插口盒',
    'B0BN8CW9ZY': 'B0BN8CW9ZY_TANMIT_(0.47-0.59)_棉芯式_1-3mm_Fine&Dot_24_牛津布_布袋',
    'B0C2L17JZB': 'B0C2L17JZB_Tesquio_(1.07-1.50)_棉芯式_0.5-5mm_Brush&Dot_8_纸盒_插口盒',
    'B0C61HTPV9': 'B0C61HTPV9_SFAIH_(0.47-0.59)_阀门式_3mm_Medium_24_PET插口盒_插口盒',
    'B0CDL915XY': 'B0CDL915XY_SFAIH_(1.07-1.50)_阀门式_0.7-3mm_Extra Fine&Medium_8_纸盒_插口盒',
    'B0CKN74CW3': 'B0CKN74CW3_BIGTHUMB_(>2.56)_阀门式_3-15mm_Jumbo_3_纸盒_插口盒',
    'B0CNYV8724': 'B0CNYV8724_TFIVE_(>2.56)_阀门式_0.7mm_Extra Fine_2_纸盒_插口盒',
    'B0CP48JPCH': 'B0CP48JPCH_ARTISTRO_(0.47-0.59)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0CQL15B98': 'B0CQL15B98_SFAIH_(0.47-0.59)_阀门式_3mm_Medium_36_PET插口盒_插口盒',
    'B0CX1BD86P': 'B0CX1BD86P_TBC The Best Crafts_(0.32-0.40)_棉芯式_1-3.9mm_Brush&Dot_24_纸盒_插口盒',
    'B0CXM7VLXV': 'B0CXM7VLXV_Droaful_(0.32-0.40)_棉芯式_1-5mm_Brush&Fine_20_纸盒_插口盒',
    'B0CYCCR82G': 'B0CYCCR82G_SRUOLOC_(>2.56)_阀门式_3-15mm_Jumbo_3_纸盒_插口盒',
    'B0CYSZ52T4': 'B0CYSZ52T4_SRUOLOC_(2.00-2.56)_阀门式_3-15mm_Jumbo_12_纸盒_插口盒',
    'B0D21NHYX8': 'B0D21NHYX8_SFAIH_(1.07-1.50)_阀门式_2-3mm_Medium_4_纸盒_插口盒',
    'B0D2H7QSX2': 'B0D2H7QSX2_Coogert_(0.47-0.59)_棉芯式_0.5-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0D471YK2F': 'B0D471YK2F_Oficrafted_(<0.32)_棉芯式_0.3mm_Medium_80_牛津布_拉链式多层布袋',
    'B0D8NGYP76': 'B0D8NGYP76_Artecho_(0.40-0.47)_棉芯式_0.5-5mm_Brush&Fine_48_纸盒_插口盒',
    'B0DFMRDMNJ': 'B0DFMRDMNJ_REALZEVA_(<0.32)_阀门式_1-3mm_Fine&Dot_24_纸盒_插口盒',
    'B0DJBS38PR': 'B0DJBS38PR_Amamao_(0.40-0.47)_直液式_1-5mm_Brush_120_纸盒_天地盖硬纸盒',
    'B0DKFMN8HF': 'B0DKFMN8HF_AKARUED_(1.07-1.50)_棉芯式_1mm_Brush&Fine_6_纸盒_插口盒',
    'B0DL9ZPMLX': 'B0DL9ZPMLX_Recheel_(<0.32)_棉芯式_1-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DLKVD4TW': 'B0DLKVD4TW_nauets_(0.47-0.59)_棉芯式_1-5mm_Fine&Dot_30_纸塑结合包装_插口盒',
    'B0DMNY5C69': 'B0DMNY5C69_TOSHARE_(0.32-0.40)_棉芯式_1-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DQX43YQD': 'B0DQX43YQD_TBC The Best Crafts_(0.32-0.40)_棉芯式_1-3.9mm_Brush&Dot_108_牛津布_拉链式多层布袋',
    'B0DRFG52NK': 'B0DRFG52NK_IVSUN_(0.47-0.59)_棉芯式_0.5-5mm_Brush&Fine_30_纸盒_插口盒',
    'B0DS1NZD89': 'B0DS1NZD89_HOTU_(0.79-1.07)_直液式_0.7-6mm_Brush_36_纸盒_天地盖硬纸盒',
    'B0F5WMP62Z': 'B0F5WMP62Z_Coogert_(0.32-0.40)_棉芯式_1-5mm_Fine&Dot_24_纸盒_插口盒',
    'B0F66RVHJW': 'B0F66RVHJW_Overseas_(0.47-0.59)_直液式_0.1-5mm_Brush_60_纸盒_插口盒'
}

# --- 3. 消费者用户画像 ---
CLASSIFICATION_RULES = {
        "User_Role": {
          '专业艺术工作者 (Professional Artist)': ['professional artist', 'pro artist', 'artist', 'illustrator', 'designer', 'comic artist', 'manga artist', 'architect', 'art studio', 'in my studio', 'commission', 'client work', 'freelance artist', 'professional work'],
          '学生 (Student)': ['student', 'school', 'college', 'university', 'art student', 'design student', 'for class', 'in my class','class notes', 'study notes', 'school project', 'assignment', 'textbook', 'studying for'],
          '教师 (Teacher)': ['teacher', 'educator', 'professor', 'art teacher', 'instructor', 'for my classroom','teaching a class', 'grading papers', 'school supplies for my students'],
          '父母 (Parent)': ['parent', 'mom', 'dad', 'mother', 'father', 'for my kids', 'for my son', 'for my daughter', 'family craft', 'homeschooling', 'with my kids'],
          '手账爱好者 (Journaler/Planner)': ['journaling', 'in my journal', 'art journal', 'junk journal', 'bible journaling','decorating my planner', 'in my planner',
                             'setting up my planner', 'planner decoration', 'planner stickers', 'planner layout','bujo', 'bullet journal',
                             'bujo spread','in my diary','scrapbooking', 'scrapbook layout', 'making a scrapbook','memory keeping', 'memory planner',
                             'hobonichi', 'leuchtturm', 'traveler\'s notebook','washi tape', 'journal stickers'
                            ],
          '业余艺术爱好者 (Hobbyist)': ['hobbyist', 'amateur artist', 'as a hobby', 'just a hobby', 'passion project', 'in my spare time', 'self-taught artist', 'just for fun', 'drawing for fun',  'painting for fun', 'my favorite pastime', 'weekend artist', 'doodling for fun','sketching in my free time', 'not a professional', 'not an artist but', 'art as a hobby'],
          '文化创意从业者 (Creative Professional)': ['creative professional', 'artisan', 'craft market', 'craft fair', 'artisan market','etsy seller', 'selling on etsy',
                                 'my etsy shop', 'small business owner', 'selling my art','content creator', 'youtuber', 'instagram artist', 'patreon creator', 'twitch streamer', 'art blogger',
                                 'workshop host', 'skillshare teacher', 'art instructor', 'leading a workshop'
                                 ],
          '初学者 (Beginner)': ['beginner', 'newbie', 'novice', 'beginner friendly', 'good for beginners', 'easy for a beginner','just starting', 'just starting out', 'getting started', 'great starting point',
                       'starter kit', 'starter set', 'my first set',  'new to art', 'new to painting', 'new to drawing', 'first time trying',
                      'learning to draw', 'learning to paint', 'easy to learn with', 'no prior experience'
                      ],          '办公人士 (Business/Office Professional)': ['for the office', 'at my office', 'office supplies', 'office work', 'at work', 'for my business','business meeting', 'work presentation', 'meeting notes', 'mind mapping for work', 'whiteboard at work',
                                  'corporate training', 'coworker', 'official report', 'signing documents', 'desk organization', 'organizing my desk'
                                 ],
          '艺术疗愈/健康追求者 (Art Therapy/Wellness Seeker)': ['art therapy', 'therapeutic', 'for relaxation', 'to relax', 'calming activity', 'for mindfulness',
                                       'helps with my anxiety', 'stress relief', 'to unwind', 'for my mental health', 'self-care activity',
                                        'peaceful activity', 'meditative drawing', 'helps me escape', 'clears my head', 'zone out',
                                        'calms me down', 'relaxing hobby', 'de-stress'
                                      ],
          '机构/批量采购者 (Institutional/Bulk Purchaser)': ['bulk order', 'bulk purchase', 'large order', 'large quantity',  'for the whole class', 'for my classroom', 'classroom set', 'school supplies order','for the office', 'office supply order', 'stocking the office', 'office set','church group', 'for the church', 'community center', 'non-profit', 'for our team', 'event supplies', 'charity donation', 'donation for',
                                      'stock up for the office', 'stock up for the classroom'],
          },
        "Gender": {
            '女性 (Female)': ['woman', 'women', 'girl', 'girls', 'she','niece','her','hers', 'wife', 'mother', 'mom', 'daughter', 'girlfriend', 'female', 'sister', 'aunt', 'grandmother', 'niece', 'lady', 'ladies'],
            '男性 (Male)': ['man', 'men', 'nephew','boy', 'boys', 'he', 'his', 'him', 'husband', 'father', 'dad', 'son', 'boyfriend', 'male', 'brother', 'uncle', 'grandfather', 'nephew', 'gentleman']
        },
        "Age_Group": {
            '儿童/幼儿': ['kid', 'kids', 'child', 'children', 'toddler', 'baby', 'preschooler', 'little one', 'grandson', 'granddaughter', 'for my son', 'for my daughter','nephew', 'niece', 'elementary school'],
            '青少年/学生': ['teen', 'teenager', 'adolescent', 'youth', 'high school', 'middle school', 'college student', 'university student', 'art student', 'for class'],
            '成年人/专业人士': ['adult', 'professional', 'pro artist', 'office work', 'at work', 'client work', 'in my studio', 'career', 'adult coloring'],
            '老年人': ['senior', 'elderly', 'retired', 'grandparent', 'grandfather', 'grandmother', 'golden years', 'grandma', 'grandpa']
        },
        "Usage": {
            '填色本填色 (Coloring Book)': ['coloring book', 'coloring books', 'adult coloring', 'colouring book', 'color page', 'coloring pages',  'adult coloring book', 'color therapy', 'mindfulness coloring', 'relaxing coloring', 'intricate coloring', 'detailed coloring', 'secret garden', 'johanna basford', 'kerby rosanes', 'hanna karlzon', 'mandalas', 'mandala coloring', 'color by number', 'mystery coloring'],
            '书法与手写艺术 (Calligraphy & Hand Lettering)': ['calligraphy', 'lettering', 'hand lettering', 'typography', 'modern calligraphy', 'brush lettering', 'faux calligraphy', 'handlettering', 'scripting', 'pointed pen', 'envelope addressing', 'flourishing', 'copperplate script', 'spencerian script', 'journal headers', 'planner headers',  'writing letters', 'place cards', 'wedding invitations'],
            '绘画创作 (Art Creation)': ['making art', 'creating art', 'for my art', 'art project', 'fine art', 'for a drawing', 'for drawing',  'illustration', 'manga', 'comic art',
                            'landscape sketch', 'urban sketching', 'artwork', 'portrait drawing',  'character design', 'sketching', 'botanical illustration', 'still life', 'figure drawing',
                            'plein air painting', 'doodling for art', 'zentangle art', 'watercolor painting', 'acrylic painting', 'inking lines', 'animal drawing', 'concept art'
            ],
            '设计工作 (Design Work)': ['design work', 'for my design work', 'professional design', 'client design', 'design project','fashion design', 'fashion illustration', 'garment design', 'textile design',
                           'product design', 'industrial design', 'product sketch', 'rendering', 'graphic design', 'logo design', 'layout design', 'branding', 'ui design', 'ux design', 'wireframing', 'mockup','architecture',
                          'architectural drawing', 'interior design', 'floor plan', 'blueprint', 'schematics','concept art', 'character design', 'storyboard', 'set design'
            ],
            '教学与学习 (Teaching & Learning)': ['art class', 'craft class', 'art school', 'for my students', 'for the class', 'classroom supplies','student work', 'school project', 'class assignment', 'grading papers', 'lesson planning', 'teaching a class', 'art education', 'homeschooling', 'learning to draw', 'learning to paint', 'skillshare class', 'online course', 'art tutorial',  'following a tutorial', 'art demonstration'],
            '手账装饰 (Journal & Planner Decoration)': ['note taking', 'taking notes', 'study notes', 'meeting notes', 'class notes', 'annotating books',  'marking up documents', 'color coding', 'color code my notes', 'organizing my calendar', 'calendar planning', 'labeling', 'making labels', 'organizing files', 'to-do list',
                                     'making lists', 'grocery list', 'keeping track of'],
            '日常记录与组织 (Daily Organization)': ['note taking', 'taking notes', 'study notes', 'meeting notes', 'class notes', 'annotating books',  'marking up documents', 'color coding', 'color code my notes', 'organizing my calendar', 'calendar planning', 'labeling', 'making labels', 'organizing files', 'to-do list',
                                  'making lists', 'grocery list', 'keeping track of'],
            '卡片与礼品制作 (Card & Gift Making)': ['card making', 'greeting card', 'handmade card', 'gift tag', 'decorating gifts',  'making cards', 'birthday card', 'christmas cards', 'thank you card', 'thank you notes', 'wedding invitations', 'anniversary card', 'valentines card', 'personalizing gifts',  'custom gifts', 'wrapping paper', 'gift wrap', 'envelope addressing', 'sentiments for cards'],
            '儿童涂鸦与早教 (Kids Activities)': ['for my kids', 'for the kids', 'with my children', 'for my toddler', 'for my son', 'for my daughter', 'kids craft', 'crafts for kids', 'family craft time', 'family fun', 'art project for kids', 'doodling', 'scribbling', 'finger painting', 'mess-free coloring','early learning', 'educational toy', 'learning colors', 'develop fine motor skills', 'preschool activities',
                               'safe for children', 'kid friendly', 'rainy day activity'],
            'DIY与手工制作 (DIY & Crafts)': ['diy project', 'craft project', 'crafting', 'for crafts', 'arts and crafts', 'handmade gifts','decorating ornaments', 'customizing shoes', 'phone case decoration', 'painting pumpkins', 'easter egg decorating', 'on glass', 'on t-shirt', 'on fabric', 'on plastic', 'on metal', 'model painting', 'miniature painting', 'painting miniatures', 'warhammer painting', 'model building',  'customizing', 'rock painting', 'mug decoration', 'wood burning', 'wood signs', 'wood crafts', 'resin art', 'resin crafts', 'polymer clay crafts', 'jewelry making', 'candle making', 'wreath making'],
            '户外与旅行创作 (Outdoor & Travel Art)': ['outdoor drawing', 'en plein air', 'urban sketching', 'travel journal', 'traveling with', 'on the go', 'field sketch','portable for travel'],
            '收藏与展示 (Collection & Display)': ['add to my collection', 'complete my collection', 'collector', 'collector\'s item', 'limited edition', 'special edition', 'collectible set', 'complete the set', 'the whole set', 'full set', 'for display', 'on my shelf'],
            '文化体验与活动 (Cultural Activities)': ['workshop', 'art event', 'cultural festival', 'live drawing', 'art therapy session', 'community art'],
            '心理疗愈 (Therapeutic Use)': ['for relaxing', 'for relaxation', 'stress relief', 'art therapy', 'therapeutic', 'calming', 'for mindfulness','emotional outlet', 'doodling to relax', 'zen', 'to unwind']
        },
        "Motivation": {
            '专业需求-色彩表现': ['high quality pigment', 'high pigment load', 'richly pigmented', 'pure pigment', 'vibrant colors', 'rich colors', 'deep saturation', 'consistent saturation', 'intense colors','lightfast', 'excellent lightfastness', 'lightfastness rating', 'archival quality', 'archival ink', 'museum quality','smooth blending', 'blends seamlessly', 'layering without getting muddy', 'excellent blendability', 'good for glazing', 'lifts cleanly', 'non-staining', 'good staining properties','true to color', 'color accuracy', 'good opacity', 'opaque coverage', 'good transparency'],
            '专业需求-性能耐用': ['pro grade', 'professional grade', 'reliable for work', 'consistent flow', 'consistent performance', 'durable tip', 'long lasting', 'for professional work', 'serious tool', 'heavy duty', 'withstand pressure', 'workhorse', 'built to last', 'daily driver', 'holds up to heavy use',  'no skipping', 'dependable performance', 'withstands abuse', 'for demanding work'],
            '基础功能需求': ['for basic use', 'for everyday use', 'for daily use', 'for school', 'for taking notes', 'gets the job done', 'does the job', 'all i need', 'nothing fancy', 'just the basics', 'no frills', 'simple and effective', 'standard use', 'for general use'],
            '艺术兴趣驱动': ['for my hobby', 'passion for art', 'spark creativity', 'express myself', 'for fun', 'artistic exploration','wanted to try', 'get back into art','for hobby'],
            '品牌信任': [ 'trusted brand', 'good reputation', 'well-known brand', 'always reliable', 'go-to brand', 'love this brand','stick with this brand', 'brand loyalty',],
            '性价比驱动': [ 'good value', 'great price', 'affordable', 'on a budget', 'good deal', 'cheap but good', 'cost effective', 'cheaper alternative'],
            '创新功能吸引': ['innovative feature', 'new feature',  'unique feature', 'special feature','new technology'],   
            '外观设计吸引': ['love the design', 'beautiful aesthetic', 'looks good', 'pretty colors', 'minimalist design', 'the look of it', 'elegant design'],
            '包装与开箱体验吸引': ['beautiful packaging', 'great unboxing experience', 'giftable', 'nice box', 'good presentation'],
            '社交驱动-口碑推荐': ['recommendation', 'recommended by', 'my friend recommended', 'my teacher recommended', 'word of mouth', 'told me to buy','saw good reviews'],
            '社交驱动-媒体影响': ['saw it on social media', 'tiktok made me buy it', 'saw it on instagram', 'youtube review', 'influencer recommended'],
            '文化与身份认同': ['culture', 'themed set', 'limited edition', 'collaboration', 'artist series', 'collectible', 'part of my identity'],
            '便携性需求': ['portable', 'on the go', 'easy to carry', 'travel set', 'compact', 'lightweight','great for travel', 'perfect for travel', 'traveling with', 'take it anywhere',  'fits in my bag', 'fits in my pocket', 'small size', 'doesn\'t take up much space','comes with a travel case', 'nice travel case'],
            '多功能性需求': ['multi-purpose', 'all-in-one', 'jack of all trades', 'works for everything', 'use it for everything', 'handles a variety of tasks', 'works on multiple surfaces','good for many things', 'one set for all my needs','many surface'],
            '礼品需求': [ 'as a gift', 'for a present', 'gift for someone', 'birthday gift', 'christmas gift', 'holiday gift','for gifts','for gift'],
            '激发创造力': ['spark creativity', 'boost creativity', 'unleash creativity', 'explore my creativity', 'creative block', 'helps with creative block', 'overcome creative block', 'new ideas',  'get the creative juices flowing', 'makes me want to create', 'inspires me to create'],
            '缓解压力与情绪调节': ['stress relief', 'for relaxation', 'to relax', 'calming', 'art therapy', 'therapeutic', 'for mindfulness', 'to unwind', 'zone out', 'peaceful activity', 'meditative', 'escape from reality', 'helps with my anxiety', 'calms me down', 'relaxing hobby'],
            '满足好奇心': ['curious about', 'wanted to see', 'heard about', 'first impression', 'wanted to try',  'give it a try', 'out of curiosity', 'just to see', 'intrigued by', 'see what the hype is about', 'first time trying', 'wanted to check it out'],
            '环保与可持续性': ['eco-friendly', 'sustainable', 'made from recycled materials', 'recycled plastic', 'recyclable', 'biodegradable', 'refillable', 'non-toxic', 'less waste', 'zero waste', 'earth-friendly',  'planet-friendly', 'good for the environment', 'environmentally friendly', 'plant-based ink', 'recyclable packaging'],
            '支持特定文化': ['support local artist', 'support local brand', 'made by local artist', 'cultural collaboration',  'artist collaboration', 'support small business',  'national pride', 'traditional craft', 'heritage brand', 'artist series', 'indie brand'],
            '追随潮流': ['trending on social media', 'all the hype', 'everyone has it', 'in style', 'viral', 'saw it on tiktok','tiktok','tiktok made me buy it', 'instagram made me buy it','instagram', 'all over youtube', 'influencer recommended', 'all the rage', 'latest trend', 'hyped up product'],
            '效率驱动': ['more efficient', 'improves efficiency', 'quick drying', 'fast drying', 'dries instantly', 'fast-drying',  'saves me time', 'saves time', 'work faster', 'speeds up my process', 'improves my workflow', 'streamline my workflow', 'cuts down on time', 'hassle-free', 'get the job done faster'],
            '学习新技能': ['learning a new skill', 'good for tutorials', 'starter kit', 'for beginners', 'beginner friendly',  'easy to learn', 'learning to draw',
                    'learning calligraphy', 'want to learn', 'just starting out',  'new to art', 'first time trying', 'for practice', 'practicing my skills',
                    'comes with instructions',  'step-by-step guide', 'improve my drawing', 'get better at painting'],
            '提升现有技能': ['challenge myself', 'advanced techniques','better tool', 'step up my game','refine my skills', 'take my art to the next level', 'more control over lines','fine-tune my work', 'mastering the craft',  'invest in my art', 'for advanced users', 'expand my capabilities', 'unlock new techniques']
        }
}

BUNDLE_PRODUCT_DIC = {
    "纸质媒介 (Paper & Pads)": {
        "黑卡纸/本": ["black paper", "black cardstock", "dark paper", "black notebook", "black pad"],
        "绘本/写生本": ["sketchbook", "sketch pad", "drawing book", "art journal", "mixed media pad"],
        "重磅马克笔纸": ["marker paper", "heavyweight paper", "smooth cardstock", "160gsm", "200gsm", "thick paper"],
        "涂鸦板/卡片": ["flashcards", "index cards", "diy cards", "tags"],
        "水彩纸/多媒体纸": ["watercolor paper", "textured paper", "cold press", "mixed media paper"],
        "黑色便利贴": ["black sticky notes", "black post-its", "dark sticky notes"]
    },
    "涂色与创作 (Coloring & Greeting)": {
        "成人涂色书": ["coloring book", "adult coloring", "mandala book", "therapy coloring"],
        "贺卡/信封": ["greeting cards", "envelopes", "invitations", "blank cards", "thank you cards"],
        "明信片": ["postcard", "postcards", "mailing cards", "blank postcards", "postal cards"], 
        "空白标签": ["gift tags", "label tags", "price tags", "hanging tags"]
    },
    "勾线与细节 (Detailing & Outlining)": {
        "极细勾线笔": ["fineliner", "micro-tip", "0.5mm pen", "ultra fine pen", "detail pen", "outline pen"],
        "铅笔/橡皮": ["graphite pencil", "pencil", "sketching pencil", "kneaded eraser", "rubber", "electric eraser"]
    },
    "表面保护 (Finishing & Protection)": {
        "亮油/保护喷雾": ["varnish", "sealer", "glossy spray", "fixative", "top coat", "clear coat"],
        "密封胶": ["sealant", "mod podge", "acrylic sealer", "glue sealer"],
        "遮蔽胶带": ["masking tape", "washi tape", "painter's tape", "decorative tape"]
    },
    "辅助与创意 (Tools & Accessories)": {
        "镂空模板": ["stencils", "drawing template", "alphabet stencil", "pattern stencil"],
        "便携笔袋/盒": ["carrying case", "storage bag", "organizer pouch", "holder", "pen stand", "acrylic holder"],
        "火漆/装饰": ["wax seal", "sealing wax", "stamps", "gold leaf"],
        "调色/混色": ["mixing palette", "paint tray", "dotting tools", "blending sponge"],
        "贴纸/胶水": ["stickers", "glue pen", "adhesive", "decals"]
    }
}

# --- 3. 数据加载函数 (已修改：统一为单一文件入口) ---
@st.cache_data
def load_raw_data():
    # === 修改核心：只读取一个名为 '常青款.xlsx' 的文件，不再区分销量和趋势 ===
    # 这里的 key 是文件名，value 是 (主类目, 子类目名称)
    # 子类目统一命名为 "全量数据"，避免页面出现分类
    data_map = {
        "常青款.xlsx": ("常青款", "全量数据")
    }
    
    combined = []
    for filename, info in data_map.items():
        if os.path.exists(filename):
            df_temp = pd.read_excel(filename)
            
            # --- 自动寻找列名 ---
            col_name = next((c for c in ['Content', 'Review Body', 'Body', 'content'] if c in df_temp.columns), df_temp.columns[0])
            asin_col = next((c for c in ['ASIN', 'Parent ASIN', 'Product ID', 'Asin', 'child_asin'] if c in df_temp.columns), None)
            
            # --- 映射逻辑 ---
            if asin_col:
                df_temp['sku_spec'] = df_temp[asin_col].astype(str).str.strip().str.upper().map(USER_CATEGORY_MAPPING).fillna("Other-Unmapped")
            else:
                df_temp['sku_spec'] = "Unknown-Spec"

            # --- 句子拆分与情感分析 ---
            df_temp = df_temp.dropna(subset=[col_name])
            
            def split_and_analyze(text):
                sentences = sent_tokenize(str(text).lower())
                results = []
                for s in sentences:
                    pol = TextBlob(s).sentiment.polarity
                    results.append({'text': s, 'polarity': pol})
                return results

            df_temp['sentences'] = df_temp[col_name].apply(split_and_analyze)
            df_exploded = df_temp.explode('sentences')
            
            # 安全提取句子内容
            df_exploded['s_text'] = df_exploded['sentences'].apply(lambda x: x['text'] if isinstance(x, dict) else "")
            df_exploded['s_pol'] = df_exploded['sentences'].apply(lambda x: x['polarity'] if isinstance(x, dict) else 0)
            
            df_exploded['main_category'] = info[0]
            df_exploded['sub_type'] = info[1]
            
            combined.append(df_exploded)
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    

# --- 4. 核心分析逻辑 (已修改：全面增加缓存，并提取全局函数以防止缓存失效) ---
@st.cache_data
def extract_advanced_features(df):
    """为每一句评论打上画像、场景、动机标签"""
    processed_df = df.copy()
    for dim_name, sub_dict in CLASSIFICATION_RULES.items():
        clean_col_name = "feat_" + dim_name
        def get_tag(text):
            text_lower = str(text).lower()
            for tag, keywords in sub_dict.items():
                if any(str(k).lower() in text_lower for k in keywords):
                    return tag
            return "未提及"
        processed_df[clean_col_name] = processed_df['s_text'].apply(get_tag)
    return processed_df

@st.cache_data
def analyze_sentiments(df_sub):
    results = []
    # 1. 获取全站真实平均分作为基准
    global_avg_rating = df_sub['Rating'].mean() if not df_sub.empty else 0
    
    for category, sub_dict in FEATURE_DIC.items():
        pos_count, neg_count = 0, 0
        hit_details = []
        all_matched_ratings = []
        
        # 准备负面模式和正面模式
        # 关键逻辑：优先处理负面标签，避免 "not good" 命中 "good"
        neg_tags = {k: v for k, v in sub_dict.items() if '负面' in k or '不满' in k}
        pos_tags = {k: v for k, v in sub_dict.items() if '正面' in k or '好评' in k}

        # --- 先跑负面匹配 ---
        neg_indices = set() # 记录负面评论的索引，防止被正面逻辑重复抓取
        for tag, keywords in neg_tags.items():
            pattern = '|'.join([re.escape(k) for k in keywords])
            mask = df_sub['s_text'].str.contains(pattern, na=False, flags=re.IGNORECASE)
            matched_df = df_sub[mask]
            
            if not matched_df.empty:
                # 原有逻辑：统计痛点提及频次（保持不变）
                neg_count += len(matched_df)
                neg_indices.update(matched_df.index.tolist())
                hit_details.append(f"{tag.split('-')[-1]}({len(matched_df)}次)")
                
                # 修改后的逻辑：仅当情感为负面/中性时，评分才参与计算
                valid_neg_mask = matched_df['s_pol'] <= 0
                all_matched_ratings.extend(matched_df[valid_neg_mask]['Rating'].tolist())

        # --- 再跑正面匹配（排除掉已经是负面的索引） ---
        remaining_df = df_sub.drop(index=list(neg_indices)) if neg_indices else df_sub
        
        for tag, keywords in pos_tags.items():
            pattern = '|'.join([re.escape(k) for k in keywords])
            mask = remaining_df['s_text'].str.contains(pattern, na=False, flags=re.IGNORECASE)
            matched_df = remaining_df[mask]
            
            if not matched_df.empty:
                # 原有逻辑：统计亮点提及频次（保持不变）
                pos_count += len(matched_df)
                
                # 修改后的逻辑：仅当情感为正面/中性时，评分才参与计算
                valid_pos_mask = matched_df['s_pol'] >= 0
                all_matched_ratings.extend(matched_df[valid_pos_mask]['Rating'].tolist())

        # 指标计算
        dim_vocal_total = pos_count + neg_count
        # 基于清洗后的评分池计算平均分
        dim_avg_rating = np.mean(all_matched_ratings) if all_matched_ratings else 0
        
        # 算法：频次 * 落差
        rating_gap = max(global_avg_rating - dim_avg_rating, 0)
        impact_index = round(neg_count * rating_gap, 2)

        results.append({
            "维度": category,
            "亮点": pos_count,
            "痛点": neg_count,
            "维度评分": round(dim_avg_rating, 2),
            "满意度": round(pos_count / dim_vocal_total * 100, 1) if dim_vocal_total > 0 else 0,
            "机会指数": impact_index,  
            "痛点分布": ", ".join(hit_details) if hit_details else "无"
        })
        
    # 同时返回 DataFrame 和 基准分，解决 NameError
    return pd.DataFrame(results).sort_values("机会指数", ascending=False), global_avg_rating

@st.cache_data
def analyze_bundle_opportunities(df_sub):
    """独立分析评论中提到的配件需求"""
    bundle_results = []
    
    for category, sub_dict in BUNDLE_PRODUCT_DIC.items():
        total_mentions = 0
        hit_details = []
        ratings_for_this_bundle = []
        
        for sub_item, keywords in sub_dict.items():
            pattern = '|'.join([re.escape(k) for k in keywords])
            # 搜索匹配的评论
            mask = df_sub['s_text'].str.contains(pattern, na=False, flags=re.IGNORECASE)
            matched_df = df_sub[mask]
            
            if not matched_df.empty:
                count = len(matched_df)
                total_mentions += count
                ratings_for_this_bundle.extend(matched_df['Rating'].tolist())
                hit_details.append(f"{sub_item}({count}次)")
        
        if total_mentions > 0:
            avg_rating = np.mean(ratings_for_this_bundle)
            bundle_results.append({
                "配件大类": category,
                "市场呼声(频次)": total_mentions,
                "关联评价均分": round(avg_rating, 2),
                "高频配件需求": " / ".join(hit_details)
            })
            
    return pd.DataFrame(bundle_results).sort_values("市场呼声(频次)", ascending=False)
    
# 【关键修复】将原先嵌套在内部的缓存函数提到全局，防止 Streamlit 每次重绘时重新定义导致缓存失效
@st.cache_data(show_spinner="正在深度分析 SKU 维度表现...")
def prepare_chart_data(data_source_id, _data_source, dims):
    d_x, d_y, d_b = dims[0], dims[1], dims[2]
    
    def get_metric_inner(target_df, dimension):
        if dimension == "其他": return 3.0, 0, 0, 0, "N/A", "N/A"
        sub_dict = FEATURE_DIC.get(dimension, {})
        if not sub_dict: return None, 0, 0, 0, "", ""
        
        pos_count, neg_count, all_matched_ratings = 0, 0, []
        hit_details, neg_texts = [], []

        for tag, keywords in sub_dict.items():
            clean_keys = [re.escape(k) for k in keywords if k.strip()]
            if not clean_keys: continue
            pat = '|'.join(clean_keys)
            mask = target_df['s_text'].str.contains(pat, na=False, flags=re.IGNORECASE)
            matched_df = target_df[mask]
            
            if not matched_df.empty:
                # --- 核心修复部分 ---
                if '负面' in tag or '不满' in tag:
                    # 负面标签：只取情感确实负面/中性的评分
                    eff_ratings = matched_df[matched_df['s_pol'] <= 0]['Rating'].tolist()
                    all_matched_ratings.extend(eff_ratings)
                    
                    neg_count += len(matched_df)
                    hit_details.append(f"{tag.split('-')[-1]}({len(matched_df)})")
                    neg_texts.extend(matched_df['s_text'].unique().tolist())
                
                elif '正面' in tag or '好评' in tag:
                    # 正面标签：只取情感确实正面/中性的评分
                    eff_ratings = matched_df[matched_df['s_pol'] >= 0]['Rating'].tolist()
                    all_matched_ratings.extend(eff_ratings)
                    
                    pos_count += len(matched_df)

        if not all_matched_ratings: return None, 0, 0, 0, "未提及", "暂无痛点原声"
        avg_score = np.mean(all_matched_ratings)
        reason = " | ".join(hit_details) if hit_details else "无明显痛点标签"
        vocal = "\n\n---\n\n".join(list(set(neg_texts))) if neg_texts else "暂无痛点原声"
        return avg_score, len(all_matched_ratings), pos_count, neg_count, reason, vocal

    plot_data = []
    all_skus = _data_source['sku_spec'].unique()
    
    for sku in all_skus:
        sku_df = _data_source[_data_source['sku_spec'] == sku]
        sc_x, cnt_x, pc_x, nc_x, re_x, vo_x = get_metric_inner(sku_df, d_x)
        sc_y, cnt_y, pc_y, nc_y, re_y, vo_y = get_metric_inner(sku_df, d_y)
        sc_b, cnt_b, pc_b, nc_b, re_b, vo_b = get_metric_inner(sku_df, d_b)
        
        if any(v is not None for v in [sc_x, sc_y, sc_b]):
            parts = str(sku).split('_')
            short_name = f"{parts[1]}-{parts[0]}" if len(parts) > 1 else str(sku)
            plot_data.append({
                'full_sku': str(sku), 'short_name': short_name,
                'score_x': sc_x or 3.0, 'score_y': sc_y or 3.0, 'score_b_val': sc_b or 3.0,
                'total_sum': (sc_x or 3.0) + (sc_y or 3.0) + (sc_b or 3.0),
                'reason_x': re_x, 'reason_y': re_y, 'reason_b': re_b,
                'vocal_x': vo_x, 'vocal_y': vo_y, 'vocal_b': vo_b,
                'pos_cnt_x': pc_x, 'neg_cnt_x': nc_x,
                'pos_cnt_y': pc_y, 'neg_cnt_y': nc_y,
                'pos_cnt_b': pc_b, 'neg_cnt_b': nc_b,
                'cnt_x': cnt_x, 'cnt_y': cnt_y, 'cnt_b': cnt_b
            })
    return pd.DataFrame(plot_data)

# 【新增】为词云生成增加缓存，词云生成耗时极大，避免每次下拉框选择时重复计算
@st.cache_data(show_spinner="生成词云中...")
def generate_wordcloud_cached(text, colormap, random_state):
    eng_stopwords = set(STOPWORDS)
    custom_garbage = {'marker', 'markers', 'pen', 'pens', 'product', 'really', 'will', 'bought', 'set', 'get', 'much', 'even', 'color', 'paint', 'colors', 'work', 'good', 'great', 'love', 'used', 'using', 'actually', 'amazon', 'br'}
    eng_stopwords.update(custom_garbage)
    wc = WordCloud(width=500, height=400, background_color='white', colormap=colormap, max_words=50, stopwords=eng_stopwords, collocations=True, random_state=random_state).generate(text)
    return wc.to_array()


# --- 5. Streamlit 页面布局 ---
st.set_page_config(page_title="丙烯笔深度调研", layout="wide")
st.title("🎨 丙烯马克笔消费者洞察看板")

df = load_raw_data()

if not df.empty:
    # 侧边栏 (由于只有一个类目，这里其实只是做个筛选的样子，实际只有一个选项)
    target = st.sidebar.radio("🎯 选择分析类目", df['main_category'].unique())
    filtered = df[df['main_category'] == target]
    
    # 此时 sub_types 应该只有一个值 ["全量数据"]
    sub_types = filtered['sub_type'].unique()

    # 遍历子类型 (这里只会循环一次)
    for sub_name in sub_types:
        # 为了美观，如果只有一种数据，可以去掉顶部的分隔空白，或者保留
        st.write("") 
        
        # 头部标题块
        st.markdown(f"""
            <div style="
                background-color: #f8f9fa; 
                padding: 20px; 
                border-radius: 15px; 
                margin-top: 20px; 
                margin-bottom: 30px; 
                border-left: 10px solid #1f77b4;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            ">
                <h2 style="
                    margin: 0; 
                    color: #1f77b4; 
                    font-size: 36px; 
                    font-weight: bold;
                ">
                    常青款深度洞察
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        sub_df = filtered[filtered['sub_type'] == sub_name]
        
        # 【关键修复】使用元组拆包接收返回值，解决 NameError
        analysis_res, global_avg_rating = analyze_sentiments(sub_df)
        
        # 顶部指标卡
        m1, m2, m3, m4 = st.columns(4)
        total_pos = analysis_res["亮点"].sum()
        total_neg = analysis_res["痛点"].sum()
        health_rate = round(total_pos / (total_pos + total_neg) * 100) if (total_pos + total_neg) > 0 else 0
        
        # 统一使用 global_avg_rating
        avg_star = round(global_avg_rating, 2)
        
        m1.metric("亮点总提及", total_pos)
        m2.metric("痛点总提及", total_neg, delta=f"-{total_neg}", delta_color="inverse")
        m3.metric("整体健康度", f"{health_rate}%")
        m4.metric("平均星级评分", f"{avg_star} ⭐")

        # 雷达图
        st.write("")
        col_radar, col_spacer = st.columns([2, 1]) 
        with col_radar:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=analysis_res['满意度'].tolist(),
                theta=analysis_res['维度'].tolist(),
                fill='toself',
                name='满意度 %',
                line_color='#3498db'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 105])),
                showlegend=False,
                title=f"维度健康度雷达图",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{sub_name}")
        
        # 柱状图 + 折线图
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(name='亮点', x=analysis_res['维度'], y=analysis_res['亮点'], 
                   marker_color='#2ecc71', text=analysis_res['亮点'], textposition='auto'),
            secondary_y=False
        )
        fig.add_trace(
            go.Bar(name='痛点', x=analysis_res['维度'], y=analysis_res['痛点'], 
                   marker_color='#e74c3c', text=analysis_res['痛点'], textposition='auto'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name='满意度 (%)', 
                x=analysis_res['维度'], 
                y=analysis_res['满意度'],
                mode='lines+markers+text', 
                text=analysis_res['满意度'].apply(lambda x: f"{x}%"), 
                textposition="top center", 
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True 
        )
        fig.add_trace(
            go.Scatter(
                name='维度评分 (1-5)', 
                x=analysis_res['维度'], 
                y=analysis_res['维度评分'],
                mode='lines+markers',
                line=dict(color='#f1c40f', width=2, dash='dot'),
                marker=dict(symbol='star', size=10)
            ),
            secondary_y=True 
        )
        
        fig.update_layout(
            title=f"各维度情感倾向分布与满意度趋势",
            barmode='group',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="提及次数", secondary_y=False)
        fig.update_yaxes(title_text="满意度分数 (%)", range=[0, 110], secondary_y=True)

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{sub_name}")

        # 竞品弱点
        st.markdown("🔍 **竞品弱点靶向追踪 (Opportunity Analysis)**")
        with st.expander("📊 如何解读机会指数？"):
            st.markdown(f"""
              **机会指数 = 痛点频次 × 评分落差** *(当前子类目全站基准分：**{avg_star}** ⭐)*
              * **计算方式**：
              1. **评分落差**：该维度的平均得分与全站基准分的差距。落差越大，代表竞品在该维度的“失分”越严重。
              2. **痛点频次**：用户对该维度负面评价的提及次数。次数越多，代表受影响的用户基数越大。
              * **结论**：指数越高，代表该维度的**“市场缺口”**越大。攻克这个痛点，就能获得最明显的口碑回升和竞争优势。
              """)
        pain_df = analysis_res.sort_values("机会指数", ascending=False).head(3)

        if not pain_df.empty:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(pain_df.iterrows()):
                with cols[idx]:
                    # 动态颜色逻辑：评分低于全站基准则为深红
                    color = "#c0392b" if row['维度评分'] < avg_star else "#d35400"
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; border-left: 8px solid {color}; 
                                 background-color: #fdfefe; border-top:1px solid #eee; border-right:1px solid #eee;
                                 box-shadow: 2px 2px 8px rgba(0,0,0,0.05); min-height: 200px;">
                        <div style="display:flex; justify-content:space-between;">
                            <h4 style="margin:0;">{row['维度']}</h4>
                            <span style="color:{color}; font-weight:bold;">得分: {row['维度评分']} ⭐</span>
                        </div>
                        <p style="color:gray; font-size:11px; margin-bottom:10px;">
                            机会指数: {row['机会指数']}
                        </p>
                        <p style="font-size:14px;"><b>核心投诉根因：</b><br/>
                        <span style="color:#2c3e50;">{row['痛点分布']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✨ 所有维度表现良好，满意度均在基准线以上！")
            
        # 词云
        st.markdown("---")
        st.markdown("### ☁️ 原声情感对比词云")
        
        pos_df = sub_df[sub_df['Rating'] >= 4.0]
        pos_text = " ".join(pos_df['s_text'].astype(str).str.lower().tolist())
        neg_df = sub_df[sub_df['Rating'] < 4.0]
        neg_text = " ".join(neg_df['s_text'].astype(str).str.lower().tolist())

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🟢 高分区 (4.0-5.0 ⭐)")
            if len(pos_text.strip()) > 30:
                # 调用带缓存的词云生成函数
                wc_pos_img = generate_wordcloud_cached(pos_text, 'Greens', 42)
                st.image(wc_pos_img, use_container_width=True)
            else:
                st.info("样本量不足")

        with col_right:
            st.subheader("🔴 低分区 (1.0-3.9 ⭐)")
            if len(neg_text.strip()) > 30:
                # 调用带缓存的词云生成函数
                wc_neg_img = generate_wordcloud_cached(neg_text, 'Reds', 24)
                st.image(wc_neg_img, use_container_width=True)
            else:
                st.success("无明显低分痛点词")

        # 原声溯源
        st.write("")
        with st.expander(f"🔍 深度探查：真实用户评价回溯"):
            target_dim = st.selectbox("选择想要探查的痛点维度:", analysis_res['维度'].tolist(), key=f"select_dim_{sub_name}")
            
            # --- 新增逻辑：提取核心投诉根因 ---
            dim_info = analysis_res[analysis_res['维度'] == target_dim].iloc[0]
            root_cause = dim_info['痛点分布'] if '痛点分布' in dim_info else "暂无根因分析"
            dim_score = dim_info['维度评分'] if '维度评分' in dim_info else 0
            
            # 这里必须定义 cause_color，否则下方 markdown 会报错
            cause_color = "#c0392b" if dim_score < 3.5 else "#d35400"
            
            # 核心投诉根因简洁版卡片
            st.markdown(f"""
                <div style="padding:12px 15px; border-radius:10px; border-left: 8px solid {cause_color}; 
                            background-color: #fff5f5; border-top:1px solid #eee; border-right:1px solid #eee;
                            margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.02);">
                    <h4 style="margin:0; color:#2c3e50; font-size:16px;">核心投诉根因</h4>
                    <p style="font-size:15px; margin-top:8px; line-height:1.5; color:#c0392b; margin-bottom:0; font-weight:500;">
                        {root_cause}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # --- 原有逻辑：原声评价列表 ---
            neg_keywords = []
            if target_dim in FEATURE_DIC:
                for tag, keys in FEATURE_DIC[target_dim].items():
                    if '负面' in tag or '不满' in tag:
                        neg_keywords.extend(keys)

            if neg_keywords:
                valid_keys = [re.escape(k) for k in neg_keywords if k.strip()]
                if not valid_keys:
                    st.info("该维度下暂无有效的负面关键词定义。")
                else:
                    search_pattern = '|'.join(valid_keys)
                    
                    # 筛选逻辑：展示匹配负面词的全部结果
                    vocal_df = sub_df[
                        (sub_df['Rating'] <= 5) & 
                        (sub_df['s_text'].str.contains(search_pattern, na=False, flags=re.IGNORECASE))
                    ].copy()
                    
                    vocal_df = vocal_df.drop_duplicates(subset=['s_text'])
                    vocal_df = vocal_df.sort_values(by='Rating', ascending=True)

                    if not vocal_df.empty:
                        total_count = len(vocal_df)
                        st.write(f"💬 **用户评价原声回溯 ({total_count} 条)：**")
                        
                        container_height = 500 if total_count > 5 else None
                        with st.container(height=container_height):
                            for i, (_, row) in enumerate(vocal_df.iterrows()):
                                text = row['s_text']
                                # 负面关键词高亮逻辑
                                sorted_keywords = sorted(list(set(neg_keywords)), key=len, reverse=True)
                                for word in sorted_keywords:
                                    if word and word.lower() in text.lower():
                                        text = re.sub(f"({re.escape(word)})", r"<span style='color:#ED4337;font-weight:bold;background-color:#FFF0F0;padding:0 2px;border-radius:2px;'>\1</span>", text, flags=re.IGNORECASE)
                                
                                st.markdown(f"**[{row['Rating']}⭐]** {text}", unsafe_allow_html=True)
                                if i < total_count - 1:
                                    st.divider()
                    else:
                        st.info("该维度下暂未匹配到对应的负面评价原声。")
            else:
                st.write("该维度暂无定义的负面关键词。")

        st.markdown("---")

        # --- 🛒 捆绑销售与配件机会分析 ---
        st.markdown("""
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
                <h3 style='margin: 0; color: #e65100;'>🎁 捆绑销售与关联购买洞察 (Bundle Insight)</h3>
                <p style='margin: 5px 0 0 0; font-size: 14px; color: #6d4c41;'>分析用户评论中自发提到的搭配产品，锁定关联销售机会。</p>
            </div>
        """, unsafe_allow_html=True)

        def analyze_bundle_opportunities(df_input):
            """独立分析函数：保留原统计逻辑，新增 quotes 存储"""
            bundle_results = []
            for category, sub_dict in BUNDLE_PRODUCT_DIC.items():
                total_mentions = 0
                hit_details = []
                ratings_list = []
                # --- 新增：记录该大类下所有的原声证据 ---
                evidence_list = [] 
                
                for sub_item, keywords in sub_dict.items():
                    pattern = '|'.join([re.escape(k) for k in keywords])
                    mask = df_input['s_text'].str.contains(pattern, na=False, flags=re.IGNORECASE)
                    matched_df = df_input[mask]
                    if not matched_df.empty:
                        count = len(matched_df)
                        total_mentions += count
                        ratings_list.extend(matched_df['Rating'].tolist())
                        hit_details.append(f"{sub_item}({count}次)")
                        
                        # --- 新增：提取该子项的 ASIN 和评论正文 ---
                        for _, row in matched_df.iterrows():
                            evidence_list.append({
                                "asin": row.get('Asin', 'N/A'),
                                "text": row['s_text'],
                                "rating": row['Rating'],
                                "tag": sub_item
                            })
                            
                if total_mentions > 0:
                    bundle_results.append({
                        "配件大类": category,
                        "市场呼声": total_mentions,
                        "关联评分": round(np.mean(ratings_list), 2) if ratings_list else 0,
                        "细节": " / ".join(hit_details),
                        "原声证据": evidence_list # 将原声存入结果集
                    })
            return pd.DataFrame(bundle_results).sort_values("市场呼声", ascending=False)

        bundle_df = analyze_bundle_opportunities(sub_df)

        if not bundle_df.empty:
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                # 侧向柱状图展示呼声 (原有功能)
                fig_bundle = go.Figure(go.Bar(
                    x=bundle_df['市场呼声'],
                    y=bundle_df['配件大类'],
                    orientation='h',
                    marker=dict(color='#ff9800', line=dict(color='#e65100', width=1))
                ))
                fig_bundle.update_layout(title="配件类别提及频次排名", height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bundle, use_container_width=True, key=f"bundle_chart_{sub_name}")
            
            with col_b2:
                st.write("📌 **核心搭配建议与原声追踪：**")
                for _, b_row in bundle_df.iterrows():
                    with st.expander(f"查看 {b_row['配件大类']} 的具体需求"):
                        # 原有功能：显示统计总结
                        st.markdown(f"**高频需求：** `{b_row['细节']}`")
                        st.markdown(f"**用户满意度：** {b_row['关联评分']} ⭐")
                        
                        # 新增功能：滚动展示具体评论和 ASIN
                        st.markdown("**🔍 关联评论原声 (ASIN 溯源):**")
                        # 只展示前 5 条最相关的原声，避免 expander 过长
                        for quote in b_row['原声证据'][:5]:
                            st.markdown(f"""
                                <div style="padding:8px; border-bottom:1px solid #eee; font-size:12px;">
                                    <span style="color:#e67e22; font-weight:bold;">[{quote['asin']}]</span> 
                                    <span style="color:#f1c40f;">{'★'*int(quote['rating'])}</span><br>
                                    <span style="color:#7f8c8d;">({quote['tag']})</span> {quote['text']}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        if b_row['关联评分'] < 4.0:
                            st.caption("💡 痛点提示：用户提及该配件时评分较低，可能是不满现有套装未包含此配件。")
        else:
            st.info("💡 当前评论样本中暂未提取到明显的配件搭配需求。")

        
        # --- 深度市场解析 ---
        # 这里的 extract_advanced_features 现在由顶部带有 @st.cache_data 的函数处理，速度极快
        sub_df = extract_advanced_features(sub_df)
        st.markdown("### 🎯 深度市场深度解析 (Advanced Market Insight)")
        st.markdown("#### 👥 用户画像分布 (Demographic Analysis)")
        
        persona_dim = st.radio("选择画像分析维度:", options=["用户身份", "性别分布", "年龄层次"], horizontal=True, key=f"persona_toggle_{sub_name}")
        dim_map = {"用户身份": "feat_User_Role", "性别分布": "feat_Gender", "年龄层次": "feat_Age_Group"}
        target_col = dim_map[persona_dim]
        
        persona_df = sub_df[(sub_df[target_col].notna()) & (sub_df[target_col] != "未提及") & (sub_df[target_col] != "Unknown")][target_col].value_counts().reset_index()

        if not persona_df.empty:
            fig_pie = go.Figure(data=[go.Pie(labels=persona_df[target_col], values=persona_df['count'], hole=.45, marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']), textinfo='percent')])
            fig_pie.update_layout(title=dict(text=f"核心访客：{persona_dim}分布", x=0.5, xanchor='center'), height=450, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            top_val = persona_df.iloc[0][target_col]
            top_pct = (persona_df.iloc[0]['count'] / persona_df['count'].sum() * 100).round(1)
            st.info(f"📊 **市场洞察：** **{top_val}** 是最主流的群体，占比高达 **{top_pct}%**。")
        else:
            st.warning(f"🔍 暂无明确的 {persona_dim} 维度数据。")

        st.markdown("---")
        st.markdown("#### 🚀 核心痛点维度评分矩阵 (Dynamic Persona-Pain Matrix)")
        
        with st.container():
            st.info("**💡 矩阵说明：** 系统已自动根据各人群数据的【机会指数】筛选其最关注的前三维度进行绘图。")

        # 1. 确定全局前三维度（作为全量数据的默认展示）
        global_top_3 = analysis_res.sort_values("机会指数", ascending=False)['维度'].tolist()[:3]
        while len(global_top_3) < 3:
            global_top_3.append("其他")

        if not analysis_res.empty:
            # =================================================================
            # 绘图展示层闭包
            # =================================================================
            def draw_sku_bubble_chart(data_source, title_label, suffix, local_dims):
                    valid_local = [d for d in local_dims if d and d != "未提及"]
                    # 优先级：人群机会指数Top3 > 全局Top3补位
                    final_dims = (valid_local + [d for d in global_top_3 if d not in valid_local])[:3]
                    d_x, d_y, d_b = final_dims[0], final_dims[1], final_dims[2]

                    # 获取绘图数据
                    data_id = f"{suffix}_{len(data_source)}" 
                    res_df = prepare_chart_data(data_id, data_source, final_dims)

                    if res_df.empty:
                            st.warning(f"⚠️ {title_label} 匹配维度下数据量过小，无法生成气泡图")
                            return

                    # --- 1. 绘制气泡图 ---
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                            x=res_df['score_x'], y=res_df['score_y'], mode='markers+text',
                            text=res_df['short_name'], textposition="top center",
                            customdata=res_df[['full_sku', 'score_b_val', 'total_sum']],
                            marker=dict(
                                    size=res_df['score_b_val'] * 12, 
                                    color=res_df['total_sum'], 
                                    colorscale='RdYlGn', showscale=True,
                                    colorbar=dict(title="综合总分"),
                                    line=dict(width=1, color='DarkSlateGrey')
                            ),
                            hovertemplate = (
                                    f"<b>%{{text}}</b><br>{d_x}: %{{x:.2f}}<br>{d_y}: %{{y:.2f}}<br>"
                                    f"{d_b}: %{{customdata[1]:.2f}}<br><b>总分: %{{customdata[2]:.2f}}</b><extra></extra>"
                            )
                    ))
                    fig.update_layout(title=f"{title_label}：维度表现矩阵 (基于机会指数Top3)", height=450, xaxis_title=f"{d_x} 评分", yaxis_title=f"{d_y} 评分")
                    st.plotly_chart(fig, use_container_width=True, key=f"bubble_{suffix}")

                    # --- 2. 交互式卡片下钻 ---
                    st.markdown(f"##### 🎯 {title_label} - 维度表现详情")
                    selected_name = st.selectbox("选择产品查看维度详情", res_df['short_name'].unique(), key=f"sel_{suffix}")
                    row = res_df[res_df['short_name'] == selected_name].iloc[0]
                    cols = st.columns(3)
                    
                    display_map = [
                            (d_x, 'score_x', 'reason_x', 'vocal_x', 'pos_cnt_x', 'neg_cnt_x'),
                            (d_y, 'score_y', 'reason_y', 'vocal_y', 'pos_cnt_y', 'neg_cnt_y'),
                            (d_b, 'score_b_val', 'reason_b', 'vocal_b', 'pos_cnt_b', 'neg_cnt_b')
                    ]

                    for i, (name, s_col, r_col, v_col, p_col, n_col) in enumerate(display_map):
                            with cols[i]:
                                    score_val = row[s_col]
                                    card_color = "#c0392b" if score_val < 3.5 else "#27ae60" if score_val > 4.2 else "#d35400"
                                    
                                    st.markdown(f"""
                                            <div style="padding:15px; border-radius:10px; border-left: 8px solid {card_color}; background-color: #fdfefe; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); min-height: 200px;">
                                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                                            <h4 style="margin:0; font-size:16px;">{name}</h4>
                                                            <span style="color:{card_color}; font-weight:bold; font-size:16px;">{score_val:.2f} ⭐</span>
                                                    </div>
                                                    <p style="font-size:11px; margin-top:5px; margin-bottom:0;">
                                                            <span style="color:#27ae60;">👍 亮点: {int(row[p_col])}</span> 
                                                            <span style="margin:0 5px; color:#ccc;">|</span> 
                                                            <span style="color:#c0392b;">👎 痛点: {int(row[n_col])}</span>
                                                    </p>
                                                    <hr style="margin:10px 0; border:0; border-top:1px solid #eee;">
                                                    <p style="font-size:13px; margin-bottom:5px;"><b>痛点分布：</b></p>
                                                    <p style="font-size:12px; color:#555; line-height:1.4;">{row[r_col]}</p>
                                            </div>
                                    """, unsafe_allow_html=True)
                                    
                                    with st.expander(f"💬 查看 {name} 痛点原声"):
                                            st.markdown(f"""
                                                    <div style='font-size:12px; color:#555; background-color:#f9f9f9; padding:10px; border-radius:5px; max-height:250px; overflow-y:auto;'>
                                                            {row[v_col]}
                                                    </div>
                                            """, unsafe_allow_html=True)
                                            
                    # --- 3. 参数明细 ---
                    with st.expander("📋 查看产品参数明细"):
                            table_rows = []
                            columns_list = ["ASIN", "Brand", "ASP用于", "出墨方式", "线宽", "笔头类型", "支数", "包装材质", "包装方式"]
                            for _, r in res_df.iterrows():
                                    parts = str(r['full_sku']).split('_')
                                    table_rows.append({col: (parts[i].strip() if i < len(parts) else "") for i, col in enumerate(columns_list)})
                            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # =================================================================
            # 渲染逻辑：核心改进部分
            # =================================================================
            top_roles = sub_df[sub_df['feat_User_Role'] != "未提及"]['feat_User_Role'].value_counts().head(3).index.tolist()
            tab_list = st.tabs(["📊 总体分析"] + [f"👤 {r}" for r in top_roles])
            
            # Tab 0: 总体
            with tab_list[0]:
                draw_sku_bubble_chart(sub_df, "全量数据", f"total_{sub_name}", global_top_3)
            
            # Tab 1-3: 不同身份人群
            for i, role in enumerate(top_roles):
                with tab_list[i+1]:
                    # A. 提取该人群子集
                    role_sub = sub_df[sub_df['feat_User_Role'] == role]
                    
                    # B. 调用分析函数计算该人群专属的统计表
                    role_analysis_res, _ = analyze_sentiments(role_sub)
                    
                    # --- 【核心修改：按样本量排序筛选维度】 ---
                    if (role_analysis_res is not None) and (not role_analysis_res.empty):
                        # 1. 在临时表中计算每个维度的总样本量 (亮点 + 痛点)
                        temp_res = role_analysis_res.copy()
                        temp_res['total_sample'] = temp_res['亮点'].fillna(0) + temp_res['痛点'].fillna(0)
                        
                        # 2. 按照总样本量降序排列，取前三个提及人数最多的维度
                        role_specific_dims = temp_res.sort_values("total_sample", ascending=False)['维度'].tolist()[:3]
                        
                        # 如果该人群维度不足3个，用全局维度补位
                        while len(role_specific_dims) < 3:
                            for gd in global_top_3:
                                if gd not in role_specific_dims:
                                    role_specific_dims.append(gd)
                                if len(role_specific_dims) == 3: break
                    else:
                        role_specific_dims = global_top_3 

                    # --- C. 数据量诊断工具 (同步更新显示) ---
                    suffix = f"role_{i}_{sub_name}" 
                    with st.expander(f"📊 样本诊断：{role} 的维度覆盖情况", expanded=False):
                        pos_col, neg_col = '亮点', '痛点'
                        
                        diag_cols = st.columns(len(role_specific_dims) + 1)
                        diag_cols[0].metric("人群总数", len(role_sub))
                        
                        for idx, d_name in enumerate(role_specific_dims):
                            if (role_analysis_res is not None) and (not role_analysis_res.empty):
                                dim_stats = role_analysis_res[role_analysis_res['维度'] == d_name]
                                
                                if not dim_stats.empty:
                                    p_val = dim_stats.iloc[0][pos_col]
                                    n_val = dim_stats.iloc[0][neg_col]
                                    total_count = int(p_val if pd.notna(p_val) else 0) + int(n_val if pd.notna(n_val) else 0)
                                    
                                    diag_cols[idx+1].metric(f"{d_name}", f"{total_count}条")
                                else:
                                    diag_cols[idx+1].metric(f"{d_name}", "0条", delta="-补位维度", delta_color="off")
                            else:
                                diag_cols[idx+1].metric(f"{d_name}", "0条")

                        if st.checkbox("查看底层统计表(调试用)", key=f"debug_{suffix}"):
                            st.dataframe(role_analysis_res if role_analysis_res is not None else "无数据")

                    # D. 绘图（此时 final_dims 内部会自动使用样本量最大的维度）
                    draw_sku_bubble_chart(role_sub, role, suffix, role_specific_dims)
































































































