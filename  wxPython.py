# coding: utf-8

import wx

class ExamplePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # 创建一些Sizer
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.GridBagSizer(hgap=5, vgap=5)    # 行和列的间距是5像素
        hSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.quote = wx.StaticText(self, label='Your quote:', pos=(20, 30))
        grid.Add(self.quote, pos=(0,0))    # 加入GridBagSizer

        self.logger = wx.TextCtrl(self, pos=(300,20), size=(200,300), style=wx.TE_MULTILINE | wx.TE_READONLY)

        self.button = wx.Button(self, label='Save', pos=(200, 325))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button)

        self.lblname = wx.StaticText(self, label='Your name:', pos=(20, 60))
        grid.Add(self.lblname, pos=(1,0))
        self.editname = wx.TextCtrl(self, value='Enter here your name:', pos=(150, 60), size=(140, -1))
        grid.Add(self.editname, pos=(1,1))
        self.Bind(wx.EVT_TEXT, self.EvtText, self.editname)
        self.Bind(wx.EVT_CHAR, self.EvtChar, self.editname)

        # 向GridBagSizer中填充空白的空间
        grid.Add((10, 40), pos=(2,0))


        self.sampleList = ['friends', 'advertising', 'web search', 'Yellow Pages']
        self.lblhear = wx.StaticText(self, label="How did you hear from us ?", pos=(20, 90))
        grid.Add(self.lblhear, pos=(3,0))
        self.edithear = wx.ComboBox(self, pos=(150, 90), size=(95, -1), choices=self.sampleList, style=wx.CB_DROPDOWN)
        grid.Add(self.edithear, pos=(3,1))
        self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox, self.edithear)
        self.Bind(wx.EVT_TEXT, self.EvtText, self.edithear)

        self.insure = wx.CheckBox(self, label="Do you want Insured Shipment ?", pos=(20,180))
        # 加入Sizer的同时，设置对齐方式和边距
        grid.Add(self.insure, pos=(4,0), span=(1,2), flag=wx.BOTTOM, border=5)
        self.Bind(wx.EVT_CHECKBOX, self.EvtCheckBox, self.insure)

        radioList = ['blue', 'red', 'yellow', 'orange', 'green', 'purple', 'navy blue', 'black', 'gray']
        self.rb = wx.RadioBox(self, label="What color would you like ?", pos=(20, 210), choices=radioList,
                              majorDimension=3, style=wx.RA_SPECIFY_COLS)
        grid.Add(self.rb, pos=(5,0), span=(1,2))    # 合并了1行2列的单元格
        self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, self.rb)

        hSizer.Add(grid, 0, wx.ALL, 5)
        hSizer.Add(self.logger)
        mainSizer.Add(hSizer, 0, wx.ALL, 5)
        mainSizer.Add(self.button, 0, wx.CENTER)
        # 可以把SetSizer()和sizer.Fit()合并成一条SetSizerAndFit()语句
        self.SetSizerAndFit(mainSizer)

    def OnClick(self, event):
        self.logger.AppendText('Click on object with Id %d\n' % event.GetId())
    def EvtText(self, event):
        self.logger.AppendText('EvtText: %s\n' % event.GetString())
    def EvtChar(self, event):
        self.logger.AppendText('EvtChar: %d\n' % event.GetKeyCode())
        event.Skip()
    def EvtComboBox(self, event):
        self.logger.AppendText('EvtComboBox: %s\n' % event.GetString())
    def EvtCheckBox(self, event):
        self.logger.AppendText('EvtCheckBox: %d\n' % event.Checked())
    def EvtRadioBox(self, event):
        self.logger.AppendText('EvtRadioBox: %d\n' % event.GetInt())


app = wx.App(False)
frame = wx.Frame(None, title="Demo with Notebook")
nb = wx.Notebook(frame)

nb.AddPage(ExamplePanel(nb), "Absolute Positioning")
nb.AddPage(ExamplePanel(nb), "Page Two")
nb.AddPage(ExamplePanel(nb), "Page Three")

frame.Show()
app.MainLoop()