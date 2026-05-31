"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { HistoryList } from "@/components/history-list";
import { AssessmentHistory } from "@/components/assessment/assessment-history";
import { useUserSettings } from "@/hooks/use-settings";

export function HistoryTabs() {
  const { t } = useUserSettings();

  return (
    <Tabs defaultValue="audio">
      <TabsList>
        <TabsTrigger value="audio">{t.history.tabAudio}</TabsTrigger>
        <TabsTrigger value="assessment">{t.history.tabAssessment}</TabsTrigger>
      </TabsList>

      <TabsContent value="audio">
        <HistoryList />
      </TabsContent>

      <TabsContent value="assessment">
        <AssessmentHistory t={t.assessment} ht={t.history} />
      </TabsContent>
    </Tabs>
  );
}
